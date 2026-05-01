from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ConnectDatabaseRequest(BaseModel):
    display_name: str
    upload_id: str


class RefreshDatabaseResponse(BaseModel):
    database_id: str
    status: str


class DatabaseConnectionResponse(BaseModel):
    database_id: str
    display_name: str
    status: str


class DatabaseSummaryResponse(BaseModel):
    database_id: str
    display_name: str
    status: str
    fingerprint: Optional[str] = None
    last_profiled_at: Optional[datetime] = None
    summary: Optional[str] = None
    warning: Optional[str] = None


class UploadResponse(BaseModel):
    upload_id: str
    file_name: str
    stored_path: str = Field(repr=False)
