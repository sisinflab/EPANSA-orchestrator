from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.services import google_auth_service
from backend.core.deps import slug_db_name
from backend.core.config import settings
from backend.services.google_utils import load_json, save_json
from backend.services.pkg_population import extract_kg_triples, delete_kg_triples


logger = logging.getLogger(__name__)

BASE_TMP_DIR = Path(settings.TMP_DIR)
SYNC_DIR = Path(settings.SYNC_DIR)
SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _google_person_to_payload(person: Dict[str, Any]):
    """
    Convert a Google People API 'person' resource into our lightweight KG payload.

    Notes:
    - Extracts display name if present.
    - Normalizes phone by removing non-digits except leading '+'.
    - Uses resourceName to build a stable contact id.
    """
    rid = person.get("resourceName") or ""
    name = (
        (person.get("names") or [{}])[0].get("displayName", "")
        if person.get("names")
        else ""
    )
    raw_phone = (
        (person.get("phoneNumbers") or [{}])[0].get("value", "")
        if person.get("phoneNumbers")
        else ""
    )

    import re as _re

    phone = _re.sub(r"[^\d+]", "", raw_phone)

    from backend.models.payloads import ContactPayload

    return ContactPayload.model_validate(
        {
            "source_app": "contacts",
            "contact": f"contact_{rid}",
            "metadata": {
                "name": name,
                "telephone_number": phone,
            },
        }
    )


async def contacts_sync(
    request: Request,
    page_size: int = 200,
    backfill: bool = False,
) -> Dict[str, Any]:
    """
    Synchronize Google Contacts (People API) for the authenticated user.

    Implementation mirrors calendar_sync structure:
    - Token initialization-only branch (no backfill, no token yet).
    - Backfill/incremental branches driven by syncToken presence.
    - Per-entity index 'contacts_processed.json' to decide insert/update/skip.
    - Deletions handled via People 'metadata.deleted'.
    - Invalid/expired syncToken (400/410) triggers a reset and early return.

    Returns a dict with summary counters and the nextSyncToken used.
    """
    logger.info("--- [CONTACTS_SYNC] Start ---")

    uid = request.app.state.user_info.get("sub")
    user_slug = slug_db_name(uid)

    user_tmp_dir = BASE_TMP_DIR / user_slug
    user_sync_dir = SYNC_DIR / user_slug
    user_tmp_dir.mkdir(parents=True, exist_ok=True)
    user_sync_dir.mkdir(parents=True, exist_ok=True)

    state_p = user_sync_dir / "contacts_state.json"
    processed_p = user_sync_dir / "contacts_processed.json"

    state = load_json(state_p, {})  # { "syncToken": ... }
    processed: Dict[str, str] = load_json(
        processed_p, {}
    )  # safe_contact_id -> etag/updateTime marker

    # Build People API client with the user's credentials.
    try:
        svc = build(
            "people",
            "v1",
            credentials=google_auth_service.get_credentials_for_user(uid),
            cache_discovery=False,
        )
    except Exception as e:
        logger.error(f"[CONTACTS] Auth failed for user {uid}: {e}", exc_info=True)
        return {"error": f"Failed to auth People API: {e}", "ok": False}

    results: Dict[str, Any] = {
        "ok": True,
        "mode": "",
        "upserts": 0,
        "deletes": 0,
        "skipped": 0,
        "errors": [],
    }

    # If no token is present, we backfill unless we are doing init-only.
    perform_backfill = backfill or not state.get("syncToken")

    # --------------------- Initialization-only branch ---------------------
    # Mirrors calendar's behavior: if no sync token and not backfilling, we just walk once to obtain nextSyncToken.
    if not state.get("syncToken") and not backfill:
        logger.info(
            "[CONTACTS] No sync token and backfill disabled. Initializing token for future incremental syncs."
        )
        try:
            init_params: Dict[str, Any] = {
                "resourceName": "people/me",
                "pageSize": page_size,
                "personFields": "metadata",
                "requestSyncToken": True,
            }
            page_token: Optional[str] = None
            next_sync: Optional[str] = None

            while True:
                if page_token:
                    init_params["pageToken"] = page_token
                else:
                    init_params.pop("pageToken", None)

                resp = svc.people().connections().list(**init_params).execute()
                page_token = resp.get("nextPageToken")
                # Accumulate the last nextSyncToken (it may only appear on the final page).
                next_sync = resp.get("nextSyncToken") or next_sync

                if not page_token:
                    break

            if next_sync:
                state["syncToken"] = next_sync
                save_json(state_p, state)

            results["mode"] = "initialized"
            logger.info(f"--- [CONTACTS_SYNC] End. Returning: {results}")
            return results
        except Exception as e:
            logger.error(
                f"[CONTACTS] Failed to initialize sync token: {e}", exc_info=True
            )
            return {"error": str(e), "ok": False}

    # --------------------- Backfill / Incremental ---------------------
    list_params: Dict[str, Any] = {
        "resourceName": "people/me",
        "pageSize": page_size,
        "personFields": "names,emailAddresses,phoneNumbers,metadata",
    }

    mode = "backfill" if perform_backfill else "incremental"
    results["mode"] = mode
    logger.info(f"[CONTACTS] Running {mode} sync for user {uid}")

    next_sync: Optional[str] = state.get("syncToken")

    if perform_backfill:
        list_params["requestSyncToken"] = True
    else:
        # if synctoken is missing, this becomes equivalent to backfill.
        if next_sync:
            list_params["syncToken"] = next_sync
        else:
            list_params["requestSyncToken"] = True

    upserts, deletes, errors = [], [], []
    page_token: Optional[str] = None

    try:
        while True:
            if page_token:
                list_params["pageToken"] = page_token
            else:
                list_params.pop("pageToken", None)

            resp = svc.people().connections().list(**list_params).execute()

            for person in resp.get("connections", []):
                rid = person.get("resourceName") or ""
                if not rid:
                    continue

                safe_id = SAFE.sub("", rid)
                json_path = user_tmp_dir / f"contact_{safe_id}.txt"

                # Deletion semantics: People API returns 'metadata.deleted' for removed contacts.
                if (person.get("metadata") or {}).get("deleted"):
                    try:
                        await delete_kg_triples(
                            fileName=f"contact_{safe_id}.txt",
                            database=request.app.state.db_name,
                            embedding_model=request.app.state.embedding_model,
                            embedding_dimension=request.app.state.embedding_dimension,
                        )
                        processed.pop(safe_id, None)  # keep local index clean
                        deletes.append(rid)
                    except Exception as ex:
                        logger.warning(
                            f"[CONTACTS] delete failed for {rid}: {ex}", exc_info=True
                        )
                        errors.append(str(ex))
                    continue

                # Change marker: prefer 'etag', otherwise use metadata.sources[0].updateTime as a fallback.
                etag_or_updated: str = person.get("etag") or ""
                if not etag_or_updated:
                    sources = (person.get("metadata") or {}).get("sources") or []
                    etag_or_updated = (
                        sources[0].get("updateTime") if sources else None
                    ) or "unknown"

                # Decide operation: insert/update/skip based on local 'processed' index.
                if safe_id not in processed:
                    op = "insert"
                else:
                    if processed.get(safe_id) == etag_or_updated:
                        results["skipped"] += 1
                        continue
                    op = "update"

                try:
                    payload = _google_person_to_payload(person)

                    # Write a temporary JSON payload file for the KG pipeline.
                    json_path.write_text(
                        payload.model_dump_json(indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    # Extract triples via embedding model configuration.
                    # embedding_model, embedding_dimension = load_embedding_model()
                    await extract_kg_triples(
                        json_path=str(json_path),
                        operation=op,
                        database=request.app.state.db_name,
                        kind="contact",
                        embedding_model=request.app.state.embedding_model,
                        embedding_dimension=request.app.state.embedding_dimension,
                    )

                    upserts.append(rid)
                    processed[safe_id] = (
                        etag_or_updated  # Update local index on success.
                    )
                except Exception as ex:
                    logger.warning(
                        f"[CONTACTS] upsert failed for {rid}: {ex}", exc_info=True
                    )
                    errors.append(str(ex))
                finally:
                    # Ensure we don't leave temp files behind.
                    json_path.unlink(missing_ok=True)

            page_token = resp.get("nextPageToken")
            next_sync = resp.get("nextSyncToken") or next_sync

            if not page_token:
                break

    except HttpError as e:
        # 400/410 -> invalid/expired sync token: reset and ask to run again.
        if e.resp is not None and e.resp.status in (400, 410):
            logger.warning(
                f"[CONTACTS] Invalid/expired sync token for user {uid}. Resetting."
            )
            state.pop("syncToken", None)
            save_json(state_p, state)
            # Persist the processed index even if token resets.
            save_json(processed_p, processed)
            return {
                "reset": True,
                "reason": "syncToken expired; run again",
                "ok": False,
            }
        # Other errors bubble up.
        logger.exception(f"[CONTACTS] HttpError: {e}")
        return {"ok": False, "error": str(e), "where": "contacts_sync.HttpError"}
    except Exception as e:
        logger.exception(f"[CONTACTS] Unhandled error: {e}")
        return {"ok": False, "error": str(e), "where": "contacts_sync"}

    # Persist tokens and indices.
    if next_sync:
        state["syncToken"] = next_sync
    save_json(state_p, state)
    save_json(processed_p, processed)

    results.update(
        {
            "upserts": len(upserts),
            "deletes": len(deletes),
            "errors": errors,
            "nextSyncToken": next_sync,
        }
    )

    logger.info(f"--- [CONTACTS_SYNC] End. Returning: {results}")
    return results
