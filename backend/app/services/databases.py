from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.db.models import ColumnProfile, DatabaseConnection, RelationshipProfile, TableProfile, User
from backend.app.db.models import PredictionRun, QueryAuditLog, SemanticMemory
from backend.app.duckdb_tools.connector import DuckDBConnector
from backend.app.duckdb_tools.profiler import DuckDBProfiler
from backend.app.memory.store import MemoryStore


class DatabaseService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.connector = DuckDBConnector()
        self.profiler = DuckDBProfiler(self.connector)
        self.memory = MemoryStore(session)

    def ensure_user(self, user_id: str) -> None:
        if self.session.get(User, user_id):
            return
        self.session.add(User(id=user_id))
        self.session.commit()

    def create_connection(self, user_id: str, display_name: str, db_path: str) -> DatabaseConnection:
        self.ensure_user(user_id)
        fingerprint = self.connector.compute_fingerprint(db_path)
        db = DatabaseConnection(
            user_id=user_id,
            display_name=display_name,
            db_uri=db_path,
            fingerprint=fingerprint,
            status="profiling",
        )
        self.session.add(db)
        self.session.commit()
        self.session.refresh(db)
        self.profile_database(db.id)
        return db

    def profile_database(self, database_id: str) -> None:
        db = self.require_database(database_id)
        bundle = self.profiler.profile_database(db.db_uri)
        self.session.execute(delete(TableProfile).where(TableProfile.database_id == database_id))
        self.session.execute(delete(ColumnProfile).where(ColumnProfile.database_id == database_id))
        self.session.execute(
            delete(RelationshipProfile).where(RelationshipProfile.database_id == database_id)
        )
        for table in bundle.tables:
            self.session.add(
                TableProfile(
                    database_id=database_id,
                    table_name=table["table_name"],
                    row_count=table["row_count"],
                    primary_key_guess=table["primary_key_guess"],
                    semantic_description=f"Table {table['table_name']} with {table['row_count']} rows.",
                    profile_json=json.dumps(table, default=str),
                )
            )
            for column in table["columns"]:
                self.session.add(
                    ColumnProfile(
                        database_id=database_id,
                        table_name=table["table_name"],
                        column_name=column["column_name"],
                        data_type=column["data_type"],
                        null_rate=column["null_rate"],
                        distinct_count=column["distinct_count"],
                        semantic_description=None,
                        profile_json=json.dumps(column, default=str),
                    )
                )
        for relationship in bundle.relationships:
            self.session.add(
                RelationshipProfile(
                    database_id=database_id,
                    from_table=relationship["from_table"],
                    from_column=relationship["from_column"],
                    to_table=relationship["to_table"],
                    to_column=relationship["to_column"],
                    confidence=relationship["confidence"],
                    evidence_json=json.dumps(relationship["evidence"]),
                )
            )

        summary = self._build_rule_based_summary(bundle.summary_input)
        self.memory.replace_memory(database_id, "database_summary", summary, bundle.summary_input)
        db.fingerprint = self.connector.compute_fingerprint(db.db_uri)
        db.status = "ready"
        db.last_profiled_at = datetime.utcnow()
        self.session.commit()

    def _build_rule_based_summary(self, summary_input: dict) -> str:
        lines = ["Database profile summary:"]
        for table in summary_input["tables"]:
            lines.append(
                f"- {table['table_name']}: {table['row_count']} rows, key guess {table['primary_key_guess'] or 'unknown'}"
            )
        if summary_input["relationships"]:
            lines.append("Likely relationships:")
            for rel in summary_input["relationships"][:10]:
                lines.append(
                    f"- {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}"
                )
        return "\n".join(lines)

    def require_database(self, database_id: str) -> DatabaseConnection:
        db = self.session.get(DatabaseConnection, database_id)
        if not db:
            raise ValueError("DATABASE_NOT_FOUND")
        return db

    def delete_database(self, database_id: str) -> None:
        self.require_database(database_id)
        self.session.execute(delete(QueryAuditLog).where(QueryAuditLog.database_id == database_id))
        self.session.execute(delete(PredictionRun).where(PredictionRun.database_id == database_id))
        self.session.execute(delete(SemanticMemory).where(SemanticMemory.database_id == database_id))
        self.session.execute(delete(RelationshipProfile).where(RelationshipProfile.database_id == database_id))
        self.session.execute(delete(ColumnProfile).where(ColumnProfile.database_id == database_id))
        self.session.execute(delete(TableProfile).where(TableProfile.database_id == database_id))
        self.session.execute(delete(DatabaseConnection).where(DatabaseConnection.id == database_id))
        self.session.commit()

    def get_database_summary(self, database_id: str) -> dict:
        db = self.require_database(database_id)
        current_fingerprint = self.connector.compute_fingerprint(db.db_uri)
        warning = None
        if db.fingerprint and db.fingerprint != current_fingerprint:
            warning = (
                "This database may have changed since it was last profiled. "
                "Predictions will use the existing profile unless you refresh."
            )
            db.status = "stale"
            self.session.commit()
        return {
            "database_id": db.id,
            "display_name": db.display_name,
            "status": db.status,
            "fingerprint": db.fingerprint,
            "last_profiled_at": db.last_profiled_at,
            "summary": self.memory.load_database_summary(database_id),
            "warning": warning,
        }

    def list_table_profiles(self, database_id: str) -> list[dict]:
        rows = self.session.scalars(
            select(TableProfile).where(TableProfile.database_id == database_id).order_by(TableProfile.table_name)
        ).all()
        return [json.loads(row.profile_json) for row in rows]

    def list_relationships(self, database_id: str) -> list[dict]:
        rows = self.session.scalars(
            select(RelationshipProfile).where(RelationshipProfile.database_id == database_id)
        ).all()
        return [
            {
                "from_table": row.from_table,
                "from_column": row.from_column,
                "to_table": row.to_table,
                "to_column": row.to_column,
                "confidence": row.confidence,
                "evidence": json.loads(row.evidence_json),
            }
            for row in rows
        ]
