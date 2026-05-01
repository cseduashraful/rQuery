from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.agent.evidence import EvidenceAccumulator
from backend.app.db.models import PredictionRun, QueryAuditLog
from backend.app.duckdb_tools.sql_safety import QueryTimeoutError, SQLSafetyLayer
from backend.app.memory.store import MemoryStore
from backend.app.services.databases import DatabaseService


class PredictionService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.memory = MemoryStore(session)
        self.safety = SQLSafetyLayer()
        self.database_service = DatabaseService(session)

    def answer_predictive_query(self, user_id: str, database_id: str, question: str) -> dict:
        db = self.database_service.require_database(database_id)
        if db.status not in {"ready", "stale"}:
            raise ValueError("DATABASE_NOT_READY")

        summary = self.memory.load_database_summary(database_id)
        memories = self.memory.search_memory(database_id, question)
        task = self._heuristic_task(question, memories)
        evidence = EvidenceAccumulator(
            user_question=question,
            database_id=database_id,
            database_summary=summary,
            task=task,
            relevant_tables=self.database_service.list_table_profiles(database_id),
            relationships=self.database_service.list_relationships(database_id),
        )

        for query in self._heuristic_queries(task, evidence.relevant_tables):
            sql = self.safety.apply_row_limit(query["sql"], 1000)
            self.safety.validate_readonly_sql(sql)
            try:
                result = self.safety.execute_safe_query(db.db_uri, sql, timeout_seconds=10)
                summary_result = self._summarize_result(result.rows)
                evidence.add_query_result(query["purpose"], sql, summary_result)
                self._log_query(user_id, database_id, sql, query["purpose"], result.row_count, "success")
            except QueryTimeoutError as exc:
                self._log_query(user_id, database_id, sql, query["purpose"], None, "timeout", str(exc))
                evidence.limitations.append("One query timed out during exploration.")
            except Exception as exc:
                self._log_query(user_id, database_id, sql, query["purpose"], None, "failed", str(exc))
                evidence.limitations.append(f"One query failed during exploration: {query['purpose']}")

        answer = self._heuristic_prediction(task, evidence)
        run = PredictionRun(
            user_id=user_id,
            database_id=database_id,
            user_question=question,
            task_json=json.dumps(task),
            evidence_packet_json=json.dumps(evidence.to_dict(), default=str),
            final_prompt="heuristic_fallback",
            answer_json=json.dumps(answer),
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        answer["prediction_run_id"] = run.id
        answer["created_at"] = run.created_at
        return answer

    def list_prediction_runs(self, database_id: str, limit: int, offset: int) -> list[PredictionRun]:
        return self.session.scalars(
            select(PredictionRun)
            .where(PredictionRun.database_id == database_id)
            .order_by(PredictionRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()

    def list_query_logs(self, database_id: str, limit: int, offset: int) -> list[QueryAuditLog]:
        return self.session.scalars(
            select(QueryAuditLog)
            .where(QueryAuditLog.database_id == database_id)
            .order_by(QueryAuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()

    def delete_prediction_run(self, run_id: str) -> None:
        run = self.session.get(PredictionRun, run_id)
        if not run:
            raise ValueError("PREDICTION_RUN_NOT_FOUND")
        self.session.execute(delete(QueryAuditLog).where(QueryAuditLog.prediction_run_id == run_id))
        self.session.delete(run)
        self.session.commit()

    def _log_query(
        self,
        user_id: str,
        database_id: str,
        sql: str,
        purpose: str,
        row_count: Optional[int],
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        self.session.add(
            QueryAuditLog(
                user_id=user_id,
                database_id=database_id,
                sql=sql,
                purpose=purpose,
                row_count=row_count,
                status=status,
                error_message=error_message,
            )
        )
        self.session.commit()

    def _heuristic_task(self, question: str, memories: list[dict]) -> dict:
        lower = question.lower()
        prediction_type = "risk_ranking" if "likely" in lower or "risk" in lower else "forecast"
        entity_name = "customer" if "customer" in lower else "entity"
        target_name = "churn" if "churn" in lower else "future outcome"
        return {
            "intent": "predictive_query",
            "prediction_type": prediction_type,
            "entity": {"name": entity_name},
            "target": {"name": target_name, "description": target_name, "time_horizon": "future"},
            "needed_information": ["activity recency", "payments", "support signals"],
            "candidate_tables": [memory["metadata"].get("table_name") for memory in memories if memory.get("metadata")],
            "assumptions": ["Heuristic mode is active because live LLM calls are disabled by default."],
        }

    def _heuristic_queries(self, task: dict, table_profiles: list[dict]) -> list[dict]:
        tables = {table["table_name"]: table for table in table_profiles}
        queries = []
        for table_name, profile in list(tables.items())[:3]:
            queries.append(
                {
                    "purpose": f"Inspect recent sample from {table_name}",
                    "sql": f'SELECT * FROM "{table_name}" LIMIT 20',
                }
            )
            temporal_columns = [
                column["column_name"]
                for column in profile["columns"]
                if column["stats"]["kind"] == "temporal"
            ]
            if temporal_columns:
                column = temporal_columns[0]
                queries.append(
                    {
                        "purpose": f"Check date range in {table_name}",
                        "sql": f'SELECT MIN("{column}") AS min_value, MAX("{column}") AS max_value, COUNT(*) AS row_count FROM "{table_name}"',
                    }
                )
        return queries[:10]

    def _summarize_result(self, rows: list[dict]) -> dict:
        if not rows:
            return {"row_count": 0, "sample_rows": []}
        return {"row_count": len(rows), "sample_rows": rows[:5]}

    def _heuristic_prediction(self, task: dict, evidence: EvidenceAccumulator) -> dict:
        confidence = "medium" if evidence.queries_run else "low"
        limitations = list(evidence.limitations)
        if not evidence.queries_run:
            limitations.append("No successful exploratory queries were available.")
        if not evidence.task.get("assumptions"):
            task["assumptions"] = ["The target outcome was inferred heuristically."]
        return {
            "answer_type": "heuristic_prediction",
            "summary": "A heuristic prediction was generated from the profiled DuckDB evidence.",
            "ranked_entities": [],
            "narrative_answer": (
                "The current implementation gathered safe exploratory evidence and produced a "
                "grounded heuristic summary. Entity-level ranking is ready for extension with "
                "LLM-guided query plans or domain-specific scoring rules."
            ),
            "limitations": limitations + task["assumptions"],
            "evidence_packet": evidence.to_dict(),
            "queries_run": evidence.queries_run,
            "confidence": confidence,
        }
