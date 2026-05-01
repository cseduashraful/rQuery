from __future__ import annotations

from backend.app.agent.prompts import build_sql_explorer_prompt
from backend.app.llm.client import LLMClient
from typing import Optional

EXPLORER_SCHEMA = {
    "type": "object",
    "properties": {
        "done": {"type": "boolean"},
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "sql": {"type": "string"},
                },
                "required": ["purpose", "sql"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["done", "queries"],
    "additionalProperties": False,
}


class SQLExplorer:
    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def next_queries(self, task: dict, schema_memory: list[dict], evidence: dict) -> dict:
        prompt = build_sql_explorer_prompt(task, schema_memory, evidence)
        return self.llm_client.generate_json("sql_explorer", prompt, EXPLORER_SCHEMA)
