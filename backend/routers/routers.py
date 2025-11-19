from __future__ import annotations

import json
import re
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, BackgroundTasks

from backend.services.pkg_population import extract_kg_triples, delete_kg_triples

from backend.core.security import jwt_dependency
from backend.core.config import settings
from backend.models.payloads import (
    AlarmPayload,
    PhoneCallPayload,
    NotePayload,
    NotePayloadLLM,
    ChatPayload,
    ChatResponse,
    PhotoPayload,
    ContactPayload,
    DeletePayload,
)
from backend.services.conversation import process_user_command
from backend.services.chat_history_service import delete_history

router = APIRouter()

SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _user_tmp_dir(request: Request) -> Path:
    """
    Resolve and ensure existence of the user's temporary directory.

    Path pattern: TMP_DIR/<user_id>
    """
    base = Path(settings.TMP_DIR)
    uid = request.app.state.user_info.get("sub")
    d = base / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json_txt(dirpath: Path, name: str, payload: dict) -> Path:
    """
    Write a JSON payload to a pretty-printed `.txt` file and return its path.
    """
    p = dirpath / name
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


async def _add_or_update_entity(p: BaseModel, request: Request, kind: str, operation: str):
    """
    Generic ingestion for text-based entities with special handling for alarms.

    Flow
    ----
    1) Build payload dict:
       - AlarmPayload -> `to_natural_language_dict()` (emits sample-accurate keys/values).
       - Others -> Pydantic model dump.
    2) Choose filename:
       - AlarmPayload recurrent -> `recurrentAlarm_<id>.txt`
       - AlarmPayload single   -> `alarm_<id>.txt`
       - Others                -> `<kind>_<id>.txt`
    3) Call `extract_kg_triples` with chosen operation.
    """
    tmp = _user_tmp_dir(request)

    # Build the payload dict
    if isinstance(p, AlarmPayload):
        payload_dict = p.to_natural_language_dict()
        # Derive filename from recurrence and id
        alarm_key = payload_dict.get("alarm", "")
        filename = f"{alarm_key}.txt" if alarm_key else f"alarm_{SAFE.sub('', p.alarm)}.txt"
    else:
        payload_dict = p.model_dump(by_alias=True)
        safe_id = SAFE.sub("", p.id) if hasattr(p, "id") else "item"
        filename = f"{kind}_{safe_id}.txt"

    json_path = _write_json_txt(tmp, filename, payload_dict)

    try:
        await extract_kg_triples(
            json_path=str(json_path),
            operation=operation,
            database=request.app.state.db_name,
            kind=kind,
            embedding_model=request.app.state.embedding_model,
            embedding_dimension=request.app.state.embedding_dimension,
        )
        json_path.unlink(missing_ok=True)
        return {"ok": True, "json_path": str(json_path)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _delete_entity(p: DeletePayload, request: Request, kind: str, embedding_model, embedding_dimension):
    """
    Generic deletion with alarm-aware filename resolution.

    For alarms:
      - If `p.id` starts with "recurrentAlarm_", delete that file name.
      - Else delete "alarm_<id>.txt".
    Others:
      - Delete "<kind>_<id>.txt".
    """
    raw_id = p.id
    if p.metadata and kind == "alarm":
        recurrence_type = p.metadata.get("recurrence_type", "")
        if not recurrence_type:
            raise ValueError("Missing 'recurrence_type' in metadata for alarm deletion.")

    safe_id = SAFE.sub("", raw_id)
    fname = f"{kind}_{safe_id}.txt"
    if kind == "alarm":
        if recurrence_type == "recurrent":
            fname = f"recurrentAlarm_{safe_id}.txt"
        elif recurrence_type == "single-occurrence":
            fname = f"alarm_{safe_id}.txt"
    else:
        fname = f"{kind}_{safe_id}.txt"

    result = await delete_kg_triples(
        fileName=fname,
        database=request.app.state.db_name,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,

    )
    return {"ok": True, "result": result}


# Alarm
@router.post("/add_alarm", status_code=201)
async def add_alarm(p: AlarmPayload, request: Request, background_tasks: BackgroundTasks, _=Depends(jwt_dependency)):
    """Insert a new alarm entry."""
    background_tasks.add_task(
        _add_or_update_entity,
        p,
        request,
        kind="alarm",
        operation="insert",
    )
    return {"ok": True, "message": f"Alarm accepted. Processing in background."}


@router.post("/update_alarm", status_code=201)
async def update_alarm(p: AlarmPayload, request: Request, background_tasks: BackgroundTasks, _=Depends(jwt_dependency)):
    """Update an existing alarm entry."""
    background_tasks.add_task(
        _add_or_update_entity,
        p,
        request,
        kind="alarm",
        operation="update",
    )
    return {"ok": True, "message": f"Update in Alarm accepted. Processing in background."}


@router.post("/delete_alarm", status_code=201)
async def delete_alarm(p: DeletePayload, request: Request, background_tasks: BackgroundTasks,
                       _=Depends(jwt_dependency)):
    """Delete an alarm entry."""
    background_tasks.add_task(
        _delete_entity,
        p,
        request,
        kind="alarm",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Deleted Alarm accepted. Processing in background."}


# Phone Call
@router.post("/add_telephone", status_code=201)
async def add_telephone(p: PhoneCallPayload, request: Request, background_tasks: BackgroundTasks,
                        _=Depends(jwt_dependency)):
    """Insert a new phone call entry."""
    background_tasks.add_task(
        _add_or_update_entity,
        p,
        request,
        kind="phoneCall",
        operation="insert",
    )
    return {"ok": True, "message": f"PhoneCall accepted. Processing in background."}


@router.post("/delete_telephone", status_code=201)
async def delete_telephone(p: DeletePayload, request: Request, background_tasks: BackgroundTasks,
                           _=Depends(jwt_dependency)):
    """Delete a phone call entry."""
    background_tasks.add_task(
        _delete_entity,
        p,
        request,
        kind="phoneCall",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Deleted PhoneCall accepted. Processing in background."}


# Note
@router.post("/add_note", status_code=201)
async def add_note(p: NotePayload, request: Request, background_tasks: BackgroundTasks, _=Depends(jwt_dependency)):
    """Insert a new note entry."""
    tmp = _user_tmp_dir(request)
    p: NotePayloadLLM = p.to_llm_payload()
    
    payload_dict = p.model_dump(by_alias=True)
    metadata_payload = p.model_dump(by_alias=True, exclude={'content'})

    safe_id = SAFE.sub("", p.note) if hasattr(p, "note") else "item"
    filename = f"note_{safe_id}.txt"
    json_path = _write_json_txt(tmp, filename, metadata_payload)

    note_content = payload_dict.get("content") or ""
    note_filename = f"noteContent_{safe_id}.txt"
    note_path = tmp / note_filename
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note_content)

    background_tasks.add_task(
        extract_kg_triples,
        json_path=str(json_path),
        unstruct_path=str(note_path),
        operation="insert",
        database=request.app.state.db_name,
        kind="note",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Note accepted. Processing in background."}


@router.post("/update_note", status_code=201)
async def update_note(p: NotePayload, request: Request, background_tasks: BackgroundTasks, _=Depends(jwt_dependency)):
    """Insert a new note entry."""
    tmp = _user_tmp_dir(request)
    p: NotePayloadLLM = p.to_llm_payload()

    payload_dict = p.model_dump(by_alias=True)
    metadata_payload = p.model_dump(by_alias=True, exclude={'content'})

    safe_id = SAFE.sub("", p.id) if hasattr(p, "id") else "item"
    filename = f"note_{safe_id}.txt"
    json_path = _write_json_txt(tmp, filename, metadata_payload)

    note_content = payload_dict.get("content") or ""
    note_filename = f"note_{safe_id}_content.txt"
    note_path = tmp / note_filename
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note_content)

    background_tasks.add_task(
        extract_kg_triples,
        json_path=str(json_path),
        unstruct_path=str(note_path),
        operation="update",
        database=request.app.state.db_name,
        kind="note",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Update in Note accepted. Processing in background."}


@router.post("/delete_note", status_code=201)
async def delete_note(p: DeletePayload, request: Request, background_tasks: BackgroundTasks, _=Depends(jwt_dependency)):
    """Delete a note entry."""
    background_tasks.add_task(
        _delete_entity,
        p,
        request,
        kind="note",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Deleted Note accepted. Processing in background."}


# Contacts
@router.post("/add_contact", status_code=201)
async def add_contact(p: ContactPayload, request: Request, background_tasks: BackgroundTasks,
                      _=Depends(jwt_dependency)):
    """Insert a new contact entry."""
    background_tasks.add_task(
        _add_or_update_entity,
        p,
        request,
        kind="contact",
        operation="insert",
    )
    return {"ok": True, "message": f"Contact accepted. Processing in background."}


@router.post("/update_contact", status_code=201)
async def update_contact(p: ContactPayload, request: Request, background_tasks: BackgroundTasks,
                         _=Depends(jwt_dependency)):
    """Update an existing contact entry."""
    background_tasks.add_task(
        _add_or_update_entity,
        p,
        request,
        kind="contact",
        operation="update",
    )
    return {"ok": True, "message": f"Update in Contact accepted. Processing in background."}


@router.post("/delete_contact", status_code=201)
async def delete_contact(p: DeletePayload, request: Request, background_tasks: BackgroundTasks,
                         _=Depends(jwt_dependency)):
    """Delete a contact entry."""
    background_tasks.add_task(
        _delete_entity,
        p,
        request,
        kind="contact",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Deleted Contact accepted. Processing in background."}


# Photo
@router.post("/upload_photo", status_code=201)
async def upload_photo(request: Request, background_tasks: BackgroundTasks, metadata: str = Form(...),
                       photo: UploadFile = File(...), _=Depends(jwt_dependency), embedding_model=None):
    """
    Upload a new photo and process it in real time.
    """
    meta = json.loads(metadata)
    p = PhotoPayload(**meta)

    tmp = _user_tmp_dir(request)
    safe_id = SAFE.sub("", p.id)

    image_name = p.metadata.filename or (photo.filename or f"photo_{safe_id}")
    image_path = tmp / image_name
    with image_path.open("wb") as out:
        while True:
            chunk = await photo.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    payload = p.model_dump(by_alias=True) | {"image_path": str(image_path)}
    json_path = _write_json_txt(tmp, f"photo_{safe_id}.txt", payload)

    try:
        background_tasks.add_task(
            extract_kg_triples,
            json_path=str(json_path),
            unstruct_path=str(image_path),
            operation="insert",
            database=request.app.state.db_name,
            kind="photo",
            embedding_model=request.app.state.embedding_model,
            embedding_dimension=request.app.state.embedding_dimension,
        )
        return {"ok": True, "image_path": str(image_path), "message": f"Photo accepted. Processing in background."}

    finally:
        try:
            json_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/delete_photo", status_code=201)
async def delete_photo(p: DeletePayload, request: Request, background_tasks: BackgroundTasks,
                       _=Depends(jwt_dependency)):
    """Delete a photo entry."""
    background_tasks.add_task(
        _delete_entity,
        p,
        request,
        kind="photo",
        embedding_model=request.app.state.embedding_model,
        embedding_dimension=request.app.state.embedding_dimension,
    )
    return {"ok": True, "message": f"Deleted Photo accepted. Processing in background."}


@router.post("/update_photo", status_code=201)
async def update_photo(request: Request, background_tasks: BackgroundTasks, metadata: str = Form(...),
                       photo: UploadFile = File(None), _=Depends(jwt_dependency), embedding_model=None):
    """
    Update photo metadata and/or replace its binary content.
    """

    meta = json.loads(metadata)
    p = PhotoPayload(**meta)
    tmp = _user_tmp_dir(request)
    safe_id = SAFE.sub("", p.id)
    image_path = None

    if photo and photo.filename:
        image_name = p.metadata.filename or (photo.filename or f"photo_{safe_id}")
        image_path = tmp / image_name
        with image_path.open("wb") as out:
            content = await photo.read()
            out.write(content)

    payload_dict = p.model_dump(by_alias=True)
    json_path = _write_json_txt(tmp, f"update_photo_{safe_id}.txt", payload_dict)

    try:
        background_tasks.add_task(
            extract_kg_triples,
            json_path=str(json_path),
            unstruct_path=str(image_path),
            operation="update",
            database=request.app.state.db_name,
            kind="photo",
            embedding_model=request.app.state.embedding_model,
            embedding_dimension=request.app.state.embedding_dimension,
        )
        message = "Photo metadata updated."
        if image_path:
            message = "Photo content and metadata updated."
        return {"ok": True, "image_path": str(image_path), "message": message}
    finally:
        try:
            json_path.unlink(missing_ok=True)
            if image_path:
                image_path.unlink(missing_ok=True)
        except Exception:
            pass


# Chat passthrough
@router.post("/chat", status_code=200)
async def handle_command(p: ChatPayload, request: Request, _=Depends(jwt_dependency)):
    """
    Handle a natural language chat command.
    """
    user_id = request.app.state.user_info.get("sub")
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        return {"ok": False, "error": "Invalid user identifier."}

    db_name = request.app.state.db_name
    return await process_user_command(user_id=user_id_int,
                                      db_name=db_name,
                                      command_text=p.text,
                                      embedding_model=request.app.state.embedding_model,
                                      )


@router.post("/reset_chat", status_code=200)
async def reset_chat(request: Request, _=Depends(jwt_dependency)):
    """
    Reset the chat history for the current user.
    """
    user_id = request.app.state.user_info.get("sub")
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        return {"ok": False, "error": "Invalid user identifier."}

    return delete_history(user_id_int)

