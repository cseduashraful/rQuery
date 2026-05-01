from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from backend.app.config import get_settings
from backend.app.db.metadata import get_db_session
from backend.app.schemas.prediction import PredictRequest, PredictResponse, PredictionRunListItem
from backend.app.services.predictions import PredictionService

router = APIRouter(prefix="/databases", tags=["predictions"])


def get_user_id(x_user_id: Optional[str] = Header(default=None)) -> str:
    return x_user_id or get_settings().default_user_id


@router.post("/{database_id}/predict", response_model=PredictResponse)
def predict(
    database_id: str,
    request: PredictRequest,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> PredictResponse:
    try:
        payload = PredictionService(session).answer_predictive_query(user_id, database_id, request.question)
    except ValueError as exc:
        code = str(exc)
        status = 404 if code == "DATABASE_NOT_FOUND" else 400
        raise HTTPException(status_code=status, detail=code) from exc
    return PredictResponse(**payload)


@router.get("/{database_id}/prediction-runs", response_model=list[PredictionRunListItem])
def list_prediction_runs(
    database_id: str,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> list[PredictionRunListItem]:
    rows = PredictionService(session).list_prediction_runs(database_id, limit, offset)
    return [
        PredictionRunListItem(
            prediction_run_id=row.id,
            question=row.user_question,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/{database_id}/query-logs")
def list_query_logs(
    database_id: str,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> list[dict]:
    rows = PredictionService(session).list_query_logs(database_id, limit, offset)
    return [
        {
            "query_log_id": row.id,
            "sql": row.sql,
            "purpose": row.purpose,
            "row_count": row.row_count,
            "status": row.status,
            "error_message": row.error_message,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@router.delete("/prediction-runs/{run_id}")
def delete_prediction_run(
    run_id: str,
    session: Session = Depends(get_db_session),
    user_id: str = Depends(get_user_id),
) -> dict:
    try:
        PredictionService(session).delete_prediction_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "prediction_run_id": run_id}
