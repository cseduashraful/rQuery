from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EvidenceAccumulator:
    user_question: str
    database_id: str
    database_summary: str
    task: dict
    relevant_tables: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    candidate_entities: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    queries_run: list[dict] = field(default_factory=list)

    def add_query_result(self, purpose: str, sql: str, summary: dict) -> None:
        self.queries_run.append({"purpose": purpose, "sql": sql, "result_summary": summary})

    def to_dict(self) -> dict:
        return asdict(self)

