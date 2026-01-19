from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
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

# -------------------------- Helpers (formatting) --------------------------

DAY_MAP = {
    "MO": "monday",
    "TU": "tuesday",
    "WE": "wednesday",
    "TH": "thursday",
    "FR": "friday",
    "SA": "saturday",
    "SU": "sunday",
}

MONTH_MAP = {
    "1": "january",
    "2": "february",
    "3": "march",
    "4": "april",
    "5": "may",
    "6": "june",
    "7": "july",
    "8": "august",
    "9": "september",
    "10": "october",
    "11": "november",
    "12": "december",
}


def _format_monthday(value: str) -> str:
    """
    Turn a BYMONTHDAY integer into a human-friendly string.
    Example: '1' -> '1st day', '-1' -> 'last day', '15' -> '15th day'.
    """
    try:
        n = int(value)
    except ValueError:
        return f"day {value}"

    if n == -1:
        return "last day"
    suffix = "th"
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        if n % 10 == 1:
            suffix = "st"
        elif n % 10 == 2:
            suffix = "nd"
        elif n % 10 == 3:
            suffix = "rd"
    return f"{n}{suffix} day"


def _ordinal(n: int) -> str:
    """
    1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', -1 -> 'last'
    """
    if n == -1:
        return "last"
    suffix = "th"
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        if n % 10 == 1:
            suffix = "st"
        elif n % 10 == 2:
            suffix = "nd"
        elif n % 10 == 3:
            suffix = "rd"
    return f"{n}{suffix}"


def _expand_byday(byday_raw: str) -> str:
    """
    Convert 'MO,WE,FR' -> 'monday, wednesday, friday'
    Handle also '1MO,-1SU' -> '1st monday, last sunday'
    """
    if not byday_raw:
        return ""
    out = []
    for token in byday_raw.split(","):
        token = token.strip().upper()
        m = re.fullmatch(r"(-?\d+)?([A-Z]{2})", token)
        if not m:
            out.append(token.lower())
            continue
        num_str, code = m.groups()
        day_name = DAY_MAP.get(code, code.lower())
        if num_str:
            try:
                n = int(num_str)
                if n == -1:
                    out.append(f"last {day_name}")
                else:
                    out.append(f"{_ordinal(n)} {day_name}")
            except ValueError:
                out.append(day_name)
        else:
            out.append(day_name)
    return ", ".join(out)


def _fmt_date_iso_to_dd_mmm_yyyy(date_iso: str) -> str:
    """
    Convert YYYY-MM-DD into 'DD Mon YYYY' (e.g., '2025-10-16' -> '16 Oct 2025').
    Returns empty string if input is empty/invalid.
    """
    if not date_iso:
        return ""
    try:
        dt = datetime.strptime(date_iso, "%Y-%m-%d")
        return dt.strftime("%d %b %Y")
    except Exception:
        return ""


def _hhmm(date_time_iso: Optional[str]) -> str:
    """
    Extract 'HH:MM' from an RFC3339 dateTime string. If None/invalid, returns ''.
    Example: '2025-10-16T09:30:00+02:00' -> '09:30'
    """
    if not date_time_iso:
        return ""
    try:
        t = date_time_iso.split("T", 1)[1]
        return t[:5]  # 'HH:MM'
    except Exception:
        return ""


def _parse_rrule_first_line(rrule_line: str) -> tuple[str, str]:
    """
    Parse a minimal subset of RRULE to a simplified (repeat_frequency, on_text) pair.
    """
    freq_map = {
        "DAILY": "daily",
        "WEEKLY": "weekly",
        "MONTHLY": "monthly",
        "YEARLY": "yearly",
    }

    freq = ""
    on_text = ""

    if not rrule_line:
        return freq, on_text

    try:
        s = rrule_line.strip()
        if s.upper().startswith("RRULE:"):
            s = s[6:]
        parts = s.split(";")
        kv = dict(p.split("=", 1) for p in parts if "=" in p)

        raw_freq = (kv.get("FREQ") or "").upper()
        freq = freq_map.get(raw_freq, raw_freq.lower())

        if "BYDAY" in kv:
            on_text = _expand_byday(kv["BYDAY"])
        elif "BYMONTHDAY" in kv:
            on_text = _format_monthday(kv["BYMONTHDAY"])
        elif "BYMONTH" in kv:
            month_num = kv["BYMONTH"].split(",")[0]
            on_text = MONTH_MAP.get(month_num, f"month {month_num}")
    except Exception:
        pass

    return freq, on_text


# -------------------------- Main sync function --------------------------


async def calendar_sync(
    request: Request,
    page_size: int = 200,
    backfill: bool = False,
) -> Dict[str, Any]:
    """
    Synchronize Google Calendar events for the authenticated user.

    New behavior (entity-specific):
    - We use a local 'events_processed.json' as a per-entity index: {safe_id: etag_or_updated}.
    - Decision rule per item:
        * Not in index        -> operation = 'insert'
        * In index with change-> operation = 'update'
        * Cancelled           -> 'delete' and remove from index
        * Unchanged           -> skip
    - We still respect syncToken paging and the 'initialize token only' path.
    """
    logger.info("--- [CALENDAR_SYNC] Start ---")

    uid = request.app.state.user_info.get("sub")
    user_slug = slug_db_name(uid)

    user_tmp_dir = BASE_TMP_DIR / user_slug
    user_sync_dir = SYNC_DIR / user_slug
    user_tmp_dir.mkdir(parents=True, exist_ok=True)
    user_sync_dir.mkdir(parents=True, exist_ok=True)

    state_p = user_sync_dir / "calendar_state.json"
    processed_p = user_sync_dir / "events_processed.json"

    state = load_json(state_p, {})
    processed: Dict[str, str] = load_json(processed_p, {})  # safe_id -> etag/updated

    # Build Calendar API service with the user's credentials
    try:
        svc = build(
            "calendar",
            "v3",
            credentials=google_auth_service.get_credentials_for_user(uid),
            cache_discovery=False,
        )
    except Exception as e:
        logger.error(f"Calendar auth failed for user {uid}: {e}", exc_info=True)
        return {"error": f"Failed to auth Calendar API: {e}", "ok": False}

    results: Dict[str, Any] = {
        "ok": True,
        "mode": "",
        "upserts": 0,
        "deletes": 0,
        "skipped": 0,
        "errors": [],
    }

    # Determine if we should do token-only initialization
    perform_backfill = backfill or not state.get("syncToken")

    # --------------------- Initialization-only branch ---------------------
    if not state.get("syncToken") and not backfill:
        logger.info(
            "[CAL] No sync token and backfill disabled. Initializing token for future incremental syncs."
        )
        try:
            init_params: Dict[str, Any] = {
                "calendarId": "primary",
                "maxResults": page_size,
                "singleEvents": False,  # keep master + exceptions
                "showDeleted": True,
            }
            page_token: Optional[str] = None
            next_sync: Optional[str] = None

            while True:
                if page_token:
                    init_params["pageToken"] = page_token
                else:
                    init_params.pop("pageToken", None)

                resp = svc.events().list(**init_params).execute()
                page_token = resp.get("nextPageToken")
                next_sync = resp.get("nextSyncToken") or next_sync

                if not page_token:
                    break

            if next_sync:
                state["syncToken"] = next_sync
                save_json(state_p, state)

            results["mode"] = "initialized"
            logger.info(f"--- [CALENDAR_SYNC] End. Returning: {results}")
            return results
        except Exception as e:
            logger.error(f"[CAL] Failed to initialize sync token: {e}", exc_info=True)
            return {"error": str(e), "ok": False}

    # --------------------- Backfill / Incremental ---------------------
    list_params: Dict[str, Any] = {
        "calendarId": "primary",
        "maxResults": page_size,
        "singleEvents": False,
        "showDeleted": True,
    }

    mode = "backfill" if perform_backfill else "incremental"
    results["mode"] = mode

    next_sync: Optional[str] = state.get("syncToken")
    if not perform_backfill and next_sync:
        list_params["syncToken"] = next_sync

    upserts, deletes, errors = [], [], []
    page_token: Optional[str] = None

    try:
        while True:
            if page_token:
                list_params["pageToken"] = page_token
            else:
                list_params.pop("pageToken", None)

            resp = svc.events().list(**list_params).execute()

            for e in resp.get("items", []):
                eid = e.get("id") or ""
                safe_id = SAFE.sub("", eid)
                json_path = user_tmp_dir / f"event_{safe_id}.txt"

                # 'Cancelled' means the event was deleted or an instance was removed.
                if e.get("status") == "cancelled":
                    try:
                        await delete_kg_triples(
                            fileName=f"event_{safe_id}.txt",
                            database=request.app.state.db_name,
                            embedding_model=request.app.state.embedding_model,
                            embedding_dimension=request.app.state.embedding_dimension,
                        )
                        processed.pop(safe_id, None)  # keep the local index clean
                        deletes.append(eid)
                    except Exception as ex:
                        logger.warning(
                            f"[CAL] delete failed for {eid}: {ex}", exc_info=True
                        )
                        errors.append(str(ex))
                    continue

                # Decide whether to insert/update/skip based on local index
                etag_or_updated = e.get("etag") or e.get("updated") or "unknown"
                if safe_id not in processed:
                    op = "insert"
                else:
                    if processed.get(safe_id) == etag_or_updated:
                        results["skipped"] += 1
                        continue
                    op = "update"

                try:
                    title = e.get("summary", "No Title")
                    start = e.get("start", {}) or {}
                    end = e.get("end", {}) or {}

                    start_date_iso = start.get("date") or (
                        start.get("dateTime", "").split("T")[0]
                        if start.get("dateTime")
                        else ""
                    )
                    start_time = _hhmm(start.get("dateTime"))
                    end_time = _hhmm(end.get("dateTime"))

                    rec_list = e.get("recurrence") or []
                    if rec_list:
                        repeat_frequency, on_text = _parse_rrule_first_line(rec_list[0])
                        payload = {
                            "source_app": "calendar",
                            "event": f"recurrentEvent_{safe_id}",
                            "recurrence_type": "recurrent",
                            "metadata": {
                                "label": title,
                                "start_time": start_time,
                                "end_time": end_time,
                                "repeat_frequency": repeat_frequency,
                                "on": on_text,
                            },
                        }
                    else:
                        payload = {
                            "source_app": "calendar",
                            "event": f"event_{safe_id}",
                            "recurrence_type": "single-occurrence",
                            "metadata": {
                                "label": title,
                                "date": _fmt_date_iso_to_dd_mmm_yyyy(start_date_iso),
                                "start_time": start_time,
                                "end_time": end_time,
                            },
                        }

                    json_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    # embedding_model, embedding_dimension = load_embedding_model()
                    await extract_kg_triples(
                        json_path=str(json_path),
                        operation=op,
                        database=request.app.state.db_name,
                        kind="event",
                        embedding_model=request.app.state.embedding_model,
                        embedding_dimension=request.app.state.embedding_dimension,
                    )
                    upserts.append(eid)
                    processed[safe_id] = (
                        etag_or_updated  # update local index on success
                    )
                except Exception as ex:
                    logger.warning(
                        f"[CAL] upsert failed for {eid}: {ex}", exc_info=True
                    )
                    errors.append(str(ex))
                finally:
                    json_path.unlink(missing_ok=True)

            page_token = resp.get("nextPageToken")
            next_sync = resp.get("nextSyncToken") or next_sync
            if not page_token:
                break

    except HttpError as e:
        if e.resp.status in (400, 410):
            logger.warning(f"[CAL] Invalid sync token for user {uid}. Resetting.")
            state.pop("syncToken", None)
            save_json(state_p, state)
            # Persist the processed index even if token resets
            save_json(processed_p, processed)
            return {
                "reset": True,
                "reason": "syncToken expired; run again",
                "ok": False,
            }
        raise

    # Persist tokens and indices
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

    logger.info(f"--- [CALENDAR_SYNC] End. Returning: {results}")
    return results
