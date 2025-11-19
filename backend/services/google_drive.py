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

from libs.llm_graph_builder.src.shared.common_fn import load_embedding_model

logger = logging.getLogger(__name__)

# Base directories for temporary payloads and persisted incremental state.
BASE_TMP_DIR = Path(settings.TMP_DIR)
SYNC_DIR = Path(settings.SYNC_DIR)

# Filename/path sanitizers to avoid unsafe characters on local FS.
SAFE_NAME = re.compile(r'[^-\w.\s]')
SAFE_ID = re.compile(r'[^A-Za-z0-9_-]')

# Google Docs/Slides MIME types that must be exported to PDF before processing.
EXPORT_TO_PDF = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
}


def _download_media(req, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return str(target_path)


def _write_meta_txt(user_dir: Path, name: str, payload: dict) -> Path:
    p = user_dir / name
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


async def process_document(
        download_req,
        unstruct_path: Path,
        meta_payload: Dict[str, Any],
        txt_filename: str,
        operation: str,
        user_dir: Path,
        db_name: str,
        results: Dict[str, Any],
) -> bool:
    """
    Download (or export) the file, write metadata .txt, then extract triples
    with the provided 'operation' (insert/update).
    """
    try:
        _download_media(download_req, unstruct_path)
        json_path = _write_meta_txt(user_dir, txt_filename, meta_payload)
        embedding_model, embedding_dimension = load_embedding_model()
        try:
            await extract_kg_triples(
                json_path=str(json_path),
                unstruct_path=str(unstruct_path),
                operation=operation,
                database=db_name,
                kind="documents",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        finally:
            json_path.unlink(missing_ok=True)
            unstruct_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        fid = meta_payload.get("id", "unknown")
        logger.warning(f"[DRIVE] Processing failed for file_id={fid}: {e}", exc_info=True)
        results["errors"].append(str(e))
        return False


def get_full_drive_path(svc, file_id):
    """
    Produce a human-friendly path-like label for a file by walking up the parents.
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


async def handle_file(
        svc,
        f: Dict[str, Any],
        operation: str,
        user_dir: Path,
        processed_versions: Dict[str, Any],
        results: Dict[str, Any],
        db_name: str,
) -> None:
    """
    Process a single Drive file (PDF or Google Doc/Slides exported to PDF)
    using the chosen operation.
    """
    fid = f["id"]
    mime = (f.get("mimeType") or "").lower()
    name = f.get("name") or f"file_{fid}"

    is_pdf = mime == "application/pdf"
    is_gdoc_pdf = mime in EXPORT_TO_PDF

    if not (is_pdf or is_gdoc_pdf):
        results["skipped"] += 1
        return

    original_path = get_full_drive_path(svc, f["id"])
    creation_date, creation_time = iso_to_date_time(f.get("createdTime"))
    modified_date, modified_time = iso_to_date_time(f.get("modifiedTime"))

    safe_file = SAFE_NAME.sub("", Path(name).name) or f"file_{fid}"
    local_path = user_dir / safe_file

    meta = {
        "source_app": "documents",
        "doc": f"doc_{fid}",
        "metadata": {
            "path": original_path,
            "creation_date": creation_date,
            "creation_time": creation_time,
            "modified_date": modified_date,
            "modified_time": modified_time,
        }
    }

    json_filename = f"doc_{SAFE_ID.sub('', fid)}.txt"
    success = False

    if is_pdf:
        req = svc.files().get_media(fileId=fid)
        success = await process_document(req, local_path, meta, json_filename, operation, user_dir, db_name, results)
    elif is_gdoc_pdf:
        req = svc.files().export_media(fileId=fid, mimeType="application/pdf")
        pdf_path = local_path.with_suffix(".pdf")
        success = await process_document(req, pdf_path, meta, json_filename, operation, user_dir, db_name, results)

    if success:
        processed_versions[fid] = f.get("version")
        results["processed"] += 1


async def drive_ingest_new(request: "Request", max_ops: int = 200, backfill: bool = False, ) -> Dict[str, Any]:

    uid = request.app.state.user_info.get("sub")
    user_slug = slug_db_name(uid)

    user_tmp_dir = BASE_TMP_DIR / user_slug
    user_sync_dir = SYNC_DIR / user_slug
    user_tmp_dir.mkdir(parents=True, exist_ok=True)
    user_sync_dir.mkdir(parents=True, exist_ok=True)

    try:
        svc = build("drive", "v3", credentials=google_auth_service.get_credentials_for_user(uid), cache_discovery=False)
    except Exception as e:
        logger.error(f"Drive auth failed for user {uid}: {e}", exc_info=True)
        return {"error": f"Drive auth failed: {e}"}

    state_p = user_sync_dir / "drive_state.json"
    proc_p = user_sync_dir / "drive_processed.json"
    state = load_json(state_p, {})
    processed_versions = load_json(proc_p, {})

    results: Dict[str, Any] = {"mode": "", "processed": 0, "deleted": 0, "skipped": 0, "errors": []}

    # No token + no backfill -> initialize token and exit
    if not state.get("startPageToken") and not backfill:
        logger.info(f"[DRIVE] No token and backfill disabled. Initializing token.")
        try:
            tok = svc.changes().getStartPageToken().execute()
            state["startPageToken"] = tok.get("startPageToken")
            save_json(state_p, state)
            results["mode"] = "initialized"
            logger.info(f"[DRIVE] Sync finished for user {uid}. Results: {results}")
            return results
        except Exception as e:
            logger.error(f"[DRIVE] Failed to initialize startPageToken: {e}", exc_info=True)
            return {"error": str(e)}

    perform_backfill = backfill or not state.get("startPageToken")

    if perform_backfill:
        results["mode"] = "initial" if not state.get("startPageToken") else "backfill_forced"
        logger.info(f"[DRIVE] Running {results['mode']} sync for user {uid}")

        if backfill:
            logger.info(f"[DRIVE] Backfill forced. Clearing processed versions cache.")
            processed_versions = {}

        page_token = None
        ops = 0
        try:
            while ops < max_ops:
                resp = svc.files().list(
                    q=("trashed=false and (mimeType='application/pdf' or mimeType contains 'vnd.google-apps.document')"),
                    pageSize=min(100, max_ops - ops),
                    fields="nextPageToken, files(id,name,mimeType,version,createdTime,modifiedTime)",
                    pageToken=page_token,
                ).execute()

                for f in resp.get("files", []):
                    if ops >= max_ops: break
                    # Decide operation from presence in processed_versions
                    op = "insert" if f["id"] not in processed_versions else "update"
                    await handle_file(svc, f, op, user_tmp_dir, processed_versions, results, request.app.state.db_name)
                    ops += 1

                page_token = resp.get("nextPageToken")
                if not page_token: break

            tok = svc.changes().getStartPageToken().execute()
            state["startPageToken"] = tok.get("startPageToken")

        except HttpError as e:
            logger.error(f"[DRIVE] HttpError during backfill: {e}")
            results["errors"].append(str(e))

    else:
        results["mode"] = "incremental"
        logger.info(f"[DRIVE] Running incremental sync for user {uid}")
        page_token = state["startPageToken"]
        ops = 0
        try:
            while page_token and ops < max_ops:
                resp = svc.changes().list(
                    pageToken=page_token,
                    fields="nextPageToken,newStartPageToken,changes(fileId,removed,file(id,name,mimeType,version,createdTime,modifiedTime))",
                ).execute()

                for ch in resp.get("changes", []):
                    if ops >= max_ops: break
                    fid = ch.get("fileId")

                    if ch.get("removed"):
                        safe_fid = SAFE_ID.sub('', fid)
                        fileName = f"doc_{safe_fid}.txt"
                        await delete_kg_triples(
                            fileName=fileName,
                            database=request.app.state.db_name,
                            embedding_model=request.app.state.embedding_model,
                            embedding_dimension=request.app.state.embedding_dimension,
                        )
                        processed_versions.pop(fid, None)  # keep local index in sync
                        results["deleted"] += 1
                        ops += 1
                        continue

                    f = ch.get("file") or {}

                    # If version didn't change, skip
                    if processed_versions.get(fid) == f.get("version"):
                        results["skipped"] += 1
                        continue

                    op = "insert" if fid not in processed_versions else "update"
                    await handle_file(svc, f, op, user_tmp_dir, processed_versions, results, request.app.state.db_name)
                    ops += 1

                page_token = resp.get("nextPageToken")
                if resp.get("newStartPageToken"):
                    state["startPageToken"] = resp["newStartPageToken"]

        except HttpError as e:
            if getattr(e, "resp", None) and e.resp.status in (400, 410):
                logger.warning(f"[DRIVE] Invalid sync token for user {uid}. Resetting.")
                state.pop("startPageToken", None)
            logger.error(f"[DRIVE] HttpError during incremental sync: {e}")
            results["errors"].append(str(e))

    save_json(state_p, state)
    save_json(proc_p, processed_versions)
    logger.info(f"[DRIVE] Sync finished for user {uid}. Results: {results}")
    return results