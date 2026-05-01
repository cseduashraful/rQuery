from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session
from typing import Optional

from backend.app.config import get_settings
from backend.app.db.metadata import get_db_session
from backend.app.schemas.database import (
    ConnectDatabaseRequest,
    DatabaseConnectionResponse,
    DatabaseSummaryResponse,
    RefreshDatabaseResponse,
    UploadResponse,
)
from backend.app.schemas.profile import DatabaseProfileResponse
from backend.app.services.databases import DatabaseService
from backend.app.services.uploads import UploadService

router = APIRouter(prefix="/databases", tags=["databases"])


def get_user_id(x_user_id: Optional[str] = Header(default=None)) -> str:
    return x_user_id or get_settings().default_user_id


@router.post("/uploads", response_model=UploadResponse)
async def upload_database(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id),
) -> UploadResponse:
    payload = await UploadService().save_upload(file, user_id)
    return UploadResponse(**payload)


@router.post("/connect", response_model=DatabaseConnectionResponse)
def connect_database(
    request: ConnectDatabaseRequest,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> DatabaseConnectionResponse:
    upload_path = UploadService().resolve_upload_path(user_id, request.upload_id)
    db = DatabaseService(session).create_connection(user_id, request.display_name, str(upload_path))
    return DatabaseConnectionResponse(database_id=db.id, display_name=db.display_name, status=db.status)


@router.get("/{database_id}", response_model=DatabaseSummaryResponse)
def get_database_summary(
    database_id: str,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> DatabaseSummaryResponse:
    try:
        payload = DatabaseService(session).get_database_summary(database_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatabaseSummaryResponse(**payload)


@router.get("/{database_id}/profile", response_model=DatabaseProfileResponse)
def get_database_profile(
    database_id: str,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> DatabaseProfileResponse:
    service = DatabaseService(session)
    db = service.require_database(database_id)
    return DatabaseProfileResponse(
        database_id=db.id,
        status=db.status,
        tables=service.list_table_profiles(database_id),
        relationships=service.list_relationships(database_id),
        summary=service.get_database_summary(database_id)["summary"],
    )


@router.post("/{database_id}/refresh", response_model=RefreshDatabaseResponse)
def refresh_database(
    database_id: str,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> RefreshDatabaseResponse:
    service = DatabaseService(session)
    service.profile_database(database_id)
    return RefreshDatabaseResponse(database_id=database_id, status="ready")


@router.delete("/{database_id}")
def delete_database(
    database_id: str,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> dict:
    service = DatabaseService(session)
    service.delete_database(database_id)
    return {"deleted": True, "database_id": database_id}
