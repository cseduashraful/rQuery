from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    question: str


class RankedEntity(BaseModel):
    entity_id: str
    rank: int
    prediction: str
    confidence: str
    reasoning: list[str]
    supporting_facts: dict


class PredictResponse(BaseModel):
    prediction_run_id: str
    answer_type: str
    summary: str
    ranked_entities: list[RankedEntity]
    narrative_answer: str
    limitations: list[str]
    evidence_packet: dict
    queries_run: list[dict]
    created_at: Optional[datetime] = None


class PredictionRunListItem(BaseModel):
    prediction_run_id: str
    question: str
    created_at: datetime
