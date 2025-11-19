from __future__ import annotations
import json, re, logging
from pathlib import Path
from typing import Any, Dict
from fastapi import Request

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from backend.services import google_auth_service
from backend.core.deps import slug_db_name
from backend.core.config import settings
from backend.services.google_utils import load_json, save_json, iso_to_date_time
from backend.services.pkg_population import extract_kg_triples, delete_kg_triples
from backend.services.img_location import analyze_image

from libs.llm_graph_builder.src.shared.common_fn import load_embedding_model

logger = logging.getLogger(__name__)

BASE_TMP_DIR = Path(settings.TMP_DIR)
SYNC_DIR = Path(settings.SYNC_DIR)

SAFE_ID = re.compile(r'[^A-Za-z0-9_-]')
SAFE_NAME = re.compile(r'[^-\w.\s]')


def _download_media(req, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return str(target_path)


def _write_meta_txt(user_dir: Path, filename: str, payload: dict) -> Path:
    path = user_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


async def process_photo(
        download_req,
        image_path: Path,
        meta_payload: Dict[str, Any],
        txt_filename: str,
        user_dir: Path,
        db_name: str,
        operation: str,
        results: Dict[str, Any],
) -> bool:
    """
    Download the image, enrich metadata (location), run extraction with the chosen operation.
    """
    try:
        _download_media(download_req, image_path)
        try:
            loc = analyze_image(image_path)
        except Exception as _:
            loc = None
        meta_payload.setdefault("metadata", {})["location"] = loc
        json_path = _write_meta_txt(user_dir, txt_filename, meta_payload)
        embedding_model, embedding_dimension = load_embedding_model()
        try:
            await extract_kg_triples(
                json_path=str(json_path),
                unstruct_path=str(image_path),
                operation=operation,
                database=db_name,
                kind="photo",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        finally:
            json_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        fid = meta_payload.get("id", "unknown")
        logger.warning(f"[PHOTOS] Processing failed for file_id={fid}: {e}", exc_info=True)
        results["errors"].append(str(e))
        return False


def get_full_drive_path(svc, file_id):
    """
    Resolve a full path-like label for the file by walking parents.
    Used only for metadata readability, not for addressing the API.
    """
    parts = []
    current_id = file_id
    try:
        while True:
            file = svc.files().get(fileId=current_id, fields="id, name, parents").execute()
            parts.insert(0, file["name"])
            parents = file.get("parents")
            if not parents:
                break
            current_id = parents[0]
        return "/" + "/".join(parts)
    except Exception as e:
        logger.warning(f"Could not resolve full drive path for {file_id}: {e}")
        return f"/unknown_path/{file_id}"


async def handle_image(
        svc,
        f: Dict[str, Any],
        operation: str,
        user_dir: Path,
        processed_versions: Dict[str, Any],
        results: Dict[str, Any],
        db_name: str,
) -> None:
    """
    Prepare metadata and run extraction for a single image using the chosen operation.
    """
    fid = f["id"]
    name = f.get("name")

    mime = (f.get("mimeType") or "").lower()
    if not mime.startswith("image/"):
        results["skipped"] += 1
        return

    original_path = get_full_drive_path(svc, f["id"])
    creation_date, creation_time = iso_to_date_time(f.get("createdTime"))
    modified_date, modified_time = iso_to_date_time(f.get("modifiedTime"))

    safe_name = SAFE_NAME.sub("", Path(name).name)
    image_path = user_dir / safe_name

    meta = {
        "source_app": "photos",
        "photo": f"photo_{fid}",
        "metadata": {
            "path": original_path,
            "creation_date": creation_date,
            "creation_time": creation_time,
            "modified_date": modified_date,
            "modified_time": modified_time,
            "location": None,
        },
    }

    txt_filename = f"photo_{SAFE_ID.sub('', fid)}.txt"
    req = svc.files().get_media(fileId=fid)

    success = await process_photo(req, image_path, meta, txt_filename, user_dir, db_name, operation, results)

    if success:
        processed_versions[fid] = f.get("version")
        results["processed"] += 1


async def photos_ingest_new(request: "Request", max_ops: int = 200, backfill: bool = False, ) -> Dict[str, Any]:

    uid = request.app.state.user_info.get("sub")
    user_slug = slug_db_name(uid)

    user_tmp_dir = BASE_TMP_DIR / user_slug
    user_sync_dir = SYNC_DIR / user_slug
    user_tmp_dir.mkdir(parents=True, exist_ok=True)
    user_sync_dir.mkdir(parents=True, exist_ok=True)

    try:
        svc = build("drive", "v3", credentials=google_auth_service.get_credentials_for_user(uid), cache_discovery=False)
    except Exception as e:
        logger.error(f"Photos/Drive auth failed for user {uid}: {e}", exc_info=True)
        return {"error": f"Auth failed: {e}"}

    state_p = user_sync_dir / "photos_state.json"
    proc_p = user_sync_dir / "photos_processed.json"
    state = load_json(state_p, {})
    processed_versions = load_json(proc_p, {})

    results: Dict[str, Any] = {"mode": "", "processed": 0, "deleted": 0, "skipped": 0, "errors": []}

    # No token + no backfill -> initialize token only
    if not state.get("startPageToken") and not backfill:
        logger.info(f"[DRIVE-PHOTO] No token and backfill disabled. Initializing token.")
        try:
            tok = svc.changes().getStartPageToken().execute()
            state["startPageToken"] = tok.get("startPageToken")
            save_json(state_p, state)
            results["mode"] = "initialized"
            logger.info(f"[DRIVE-PHOTO] Sync finished for user {uid}. Results: {results}")
            return results
        except Exception as e:
            logger.error(f"[DRIVE-PHOTO] Failed to initialize startPageToken: {e}", exc_info=True)
            return {"error": str(e)}

    perform_backfill = backfill or not state.get("startPageToken")

    if perform_backfill:
        results["mode"] = "initial" if not state.get("startPageToken") else "backfill_forced"
        logger.info(f"[PHOTOS] Running {results['mode']} sync for user {uid}")

        if backfill:
            logger.info(f"[PHOTOS] Backfill forced. Clearing processed versions cache.")
            processed_versions = {}

        page, ops = None, 0
        try:
            while ops < max_ops:
                resp = svc.files().list(
                    q="mimeType contains 'image/' and trashed=false",
                    pageSize=min(100, max_ops - ops),
                    fields="nextPageToken, files(id,name,mimeType,version,createdTime,modifiedTime)",
                    pageToken=page,
                ).execute()

                for f in resp.get("files", []):
                    if ops >= max_ops: break
                    # Decide op from processed_versions even during backfill
                    op = "insert" if f["id"] not in processed_versions else "update"
                    await handle_image(svc, f, op, user_tmp_dir, processed_versions, results, request.app.state.db_name)
                    ops += 1

                page = resp.get("nextPageToken")
                if not page: break

            tok = svc.changes().getStartPageToken().execute()
            state["startPageToken"] = tok.get("startPageToken")

        except HttpError as e:
            logger.error(f"[PHOTOS] HttpError during backfill: {e}")
            results["errors"].append(str(e))

    else:
        results["mode"] = "incremental"
        logger.info(f"[PHOTOS] Running incremental sync for user {uid}")
        page = state["startPageToken"]
        ops = 0
        try:
            while page and ops < max_ops:
                resp = svc.changes().list(
                    pageToken=page,
                    fields="nextPageToken,newStartPageToken,changes(fileId,removed,file(id,name,mimeType,version,createdTime,modifiedTime))",
                ).execute()

                for ch in resp.get("changes", []):
                    if ops >= max_ops: break
                    fid = ch.get("fileId")

                    if ch.get("removed"):
                        await delete_kg_triples(fileName=f"photo_{SAFE_ID.sub('', fid)}.txt",
                                                database=request.app.state.db_name,
                                                embedding_model=request.app.state.embedding_model,
                                                embedding_dimension=request.app.state.embedding_dimension,
                                                )
                        processed_versions.pop(fid, None)  # keep cache in sync with deletions
                        results["deleted"] += 1
                        ops += 1
                        continue

                    f = ch.get("file") or {}
                    if not (f.get("mimeType") or "").lower().startswith("image/"):
                        results["skipped"] += 1
                        continue

                    # Skip if version unchanged; else decide op as update (the item exists)
                    if processed_versions.get(fid) == f.get("version"):
                        results["skipped"] += 1
                        continue

                    op = "insert" if fid not in processed_versions else "update"
                    await handle_image(svc, f, op, user_tmp_dir, processed_versions, results, request.app.state.db_name)
                    ops += 1

                page = resp.get("nextPageToken")
                if resp.get("newStartPageToken"):
                    state["startPageToken"] = resp["newStartPageToken"]

        except HttpError as e:
            if getattr(e, "resp", None) and e.resp.status in (400, 410):
                logger.warning(f"[PHOTOS] Invalid sync token for user {uid}. Resetting.")
                state.pop("startPageToken", None)
            logger.error(f"[PHOTOS] HttpError during incremental sync: {e}")
            results["errors"].append(str(e))

    save_json(state_p, state)
    save_json(proc_p, processed_versions)
    logger.info(f"[PHOTOS] Sync finished for user {uid}. Results: {results}")
    return results
