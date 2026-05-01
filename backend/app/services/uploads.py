from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.app.config import get_settings


class UploadService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def save_upload(self, upload: UploadFile, user_id: str) -> dict:
        upload_id = f"uploaded_file_{uuid4().hex[:12]}"
        suffix = Path(upload.filename or "database.duckdb").suffix or ".duckdb"
        user_dir = self.settings.upload_root / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        dest = user_dir / f"{upload_id}{suffix}"
        content = await upload.read()
        dest.write_bytes(content)
        return {"upload_id": upload_id, "file_name": upload.filename or dest.name, "stored_path": str(dest)}

    def resolve_upload_path(self, user_id: str, upload_id: str) -> Path:
        user_dir = self.settings.upload_root / user_id
        matches = list(user_dir.glob(f"{upload_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Upload not found: {upload_id}")
        return matches[0]

