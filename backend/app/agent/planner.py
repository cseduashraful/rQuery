from __future__ import annotations

from backend.app.agent.prompts import build_task_planning_prompt
from backend.app.llm.client import LLMClient
from typing import Optional

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "prediction_type": {"type": "string"},
        "entity": {"type": "object"},
        "target": {"type": "object"},
        "needed_information": {"type": "array", "items": {"type": "string"}},
        "candidate_tables": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "intent",
        "prediction_type",
        "entity",
        "target",
        "needed_information",
        "candidate_tables",
        "assumptions",
    ],
    "additionalProperties": True,
}


class TaskPlanner:
    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def plan(self, question: str, db_summary: str, memories: list[dict]) -> dict:
        prompt = build_task_planning_prompt(question, db_summary, memories)
        return self.llm_client.generate_json("task_planner", prompt, TASK_SCHEMA)
