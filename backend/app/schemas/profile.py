from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class ColumnProfileSchema(BaseModel):
    column_name: str
    data_type: str
    null_rate: Optional[float] = None
    distinct_count: Optional[int] = None
    stats: dict


class TableProfileSchema(BaseModel):
    table_name: str
    row_count: int
    primary_key_guess: Optional[str] = None
    columns: list[ColumnProfileSchema]
    sample_rows: list[dict]


class DatabaseProfileResponse(BaseModel):
    database_id: str
    status: str
    tables: list[TableProfileSchema]
    relationships: list[dict]
    summary: Optional[str] = None
