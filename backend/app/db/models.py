from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.metadata import Base


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class DatabaseConnection(Base, TimestampMixin):
    __tablename__ = "database_connections"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("db"))
    user_id: Mapped[str] = mapped_column(Text, ForeignKey("users.id"), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    db_type: Mapped[str] = mapped_column(Text, default="duckdb", nullable=False)
    db_uri: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="profiling")
    last_profiled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class TableProfile(Base, TimestampMixin):
    __tablename__ = "table_profiles"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("tbl"))
    database_id: Mapped[str] = mapped_column(
        Text, ForeignKey("database_connections.id"), nullable=False, index=True
    )
    table_name: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    primary_key_guess: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    semantic_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    profile_json: Mapped[str] = mapped_column(Text, nullable=False)


class ColumnProfile(Base, TimestampMixin):
    __tablename__ = "column_profiles"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("col"))
    database_id: Mapped[str] = mapped_column(
        Text, ForeignKey("database_connections.id"), nullable=False, index=True
    )
    table_name: Mapped[str] = mapped_column(Text, nullable=False)
    column_name: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    null_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distinct_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    semantic_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    profile_json: Mapped[str] = mapped_column(Text, nullable=False)


class RelationshipProfile(Base, TimestampMixin):
    __tablename__ = "relationship_profiles"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("rel"))
    database_id: Mapped[str] = mapped_column(
        Text, ForeignKey("database_connections.id"), nullable=False, index=True
    )
    from_table: Mapped[str] = mapped_column(Text, nullable=False)
    from_column: Mapped[str] = mapped_column(Text, nullable=False)
    to_table: Mapped[str] = mapped_column(Text, nullable=False)
    to_column: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False)


class SemanticMemory(Base, TimestampMixin):
    __tablename__ = "semantic_memories"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("mem"))
    database_id: Mapped[str] = mapped_column(
        Text, ForeignKey("database_connections.id"), nullable=False, index=True
    )
    memory_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("run"))
    user_id: Mapped[str] = mapped_column(Text, ForeignKey("users.id"), nullable=False, index=True)
    database_id: Mapped[str] = mapped_column(
        Text, ForeignKey("database_connections.id"), nullable=False, index=True
    )
    user_question: Mapped[str] = mapped_column(Text, nullable=False)
    task_json: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_packet_json: Mapped[str] = mapped_column(Text, nullable=False)
    final_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    answer_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class QueryAuditLog(Base):
    __tablename__ = "query_audit_logs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: gen_id("qry"))
    user_id: Mapped[str] = mapped_column(Text, ForeignKey("users.id"), nullable=False, index=True)
    database_id: Mapped[str] = mapped_column(
        Text, ForeignKey("database_connections.id"), nullable=False, index=True
    )
    prediction_run_id: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("prediction_runs.id"), nullable=True
    )
    sql: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
