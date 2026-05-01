from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class ErrorDetails(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: ErrorDetails
