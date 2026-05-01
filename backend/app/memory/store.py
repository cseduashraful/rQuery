from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.db.models import SemanticMemory


class MemoryStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_memory(
        self,
        database_id: str,
        memory_type: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self.session.execute(
            delete(SemanticMemory).where(
                SemanticMemory.database_id == database_id,
                SemanticMemory.memory_type == memory_type,
            )
        )
        self.session.add(
            SemanticMemory(
                database_id=database_id,
                memory_type=memory_type,
                content=content,
                metadata_json=json.dumps(metadata or {}),
            )
        )

    def search_memory(self, database_id: str, user_query: str) -> list[dict]:
        rows = self.session.scalars(
            select(SemanticMemory).where(SemanticMemory.database_id == database_id)
        ).all()
        terms = {part.strip().lower() for part in user_query.split() if part.strip()}
        ranked = []
        for row in rows:
            haystack = f"{row.memory_type} {row.content}".lower()
            score = sum(term in haystack for term in terms)
            ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "memory_type": row.memory_type,
                "content": row.content,
                "metadata": json.loads(row.metadata_json or "{}"),
            }
            for score, row in ranked[:10]
            if score > 0 or row.memory_type == "database_summary"
        ]

    def load_database_summary(self, database_id: str) -> str:
        row = self.session.scalars(
            select(SemanticMemory).where(
                SemanticMemory.database_id == database_id,
                SemanticMemory.memory_type == "database_summary",
            )
        ).first()
        return row.content if row else ""
