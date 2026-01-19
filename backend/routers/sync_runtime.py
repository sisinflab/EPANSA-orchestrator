from __future__ import annotations
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, Request, HTTPException

from backend.core.security import jwt_dependency  # keeps admin-only access if you want
from backend.services.google_contacts import contacts_sync
from backend.services.google_calendar import calendar_sync
from backend.services.google_drive import drive_ingest_new
from backend.services.google_photos import photos_ingest_new
from backend.services.secure_store import credentials_service
from backend.core.deps import slug_db_name

import asyncio

router = APIRouter(prefix="/sync", tags=["Sync"])


async def _run_all(request: Request, backfill: bool) -> Dict[str, Any]:
    """Run all Google sync services for the current user."""
    results = await asyncio.gather(
        contacts_sync(request, backfill=backfill),
        calendar_sync(request, backfill=backfill),
        drive_ingest_new(request, backfill=backfill),
        photos_ingest_new(request, backfill=backfill),
        return_exceptions=True,  # if one fail don't stop the others
    )
    service_names = ["contacts", "calendar", "drive", "photos"]
    return {name: res for name, res in zip(service_names, results)}


@router.post("/run_for_me")
async def run_for_me(request: Request, backfill: bool, _=Depends(jwt_dependency)):
    """Trigger runtime sync for the authenticated user."""
    return await _run_all(request, backfill)


@router.post("/run_for_user")
async def run_for_user(
    user_id: str, request: Request, backfill: bool, _=Depends(jwt_dependency)
):
    """Trigger runtime sync for a specific user id (admin use)."""
    if not user_id:
        raise HTTPException(400, "user_id is required")
    # Minimal tenant context for the services:
    request.app.state.user_info = {"sub": user_id}
    return await _run_all(request, backfill)


@router.post("/run_for_all_users")
async def run_for_all_users(
    request: Request, backfill: bool = False, _=Depends(jwt_dependency)
):
    """Trigger runtime sync for all users with stored Google credentials."""
    user_ids: List[str] = credentials_service.get_all_user_ids()
    results: Dict[str, Any] = {}
    for uid in user_ids:
        temp_request = Request(scope=request.scope)
        temp_request.app.state.user_info = {"sub": uid}
        temp_request.app.state.db_name = slug_db_name(uid)  # ensure db_name is set

        results[uid] = await _run_all(temp_request, backfill)
    return {"ok": True, "count": len(user_ids), "results": results}
