from __future__ import annotations

from backend.app.agent.prompts import build_prediction_prompt
from backend.app.llm.client import LLMClient
from typing import Optional

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_type": {"type": "string"},
        "summary": {"type": "string"},
        "ranked_entities": {"type": "array", "items": {"type": "object"}},
        "narrative_answer": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer_type", "summary", "ranked_entities", "narrative_answer", "limitations"],
    "additionalProperties": True,
}


class FinalPredictor:
    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def predict(self, evidence_packet: dict) -> tuple[str, dict]:
        prompt = build_prediction_prompt(evidence_packet)
        result = self.llm_client.generate_json("final_predictor", prompt, PREDICTION_SCHEMA)
        return prompt, result
