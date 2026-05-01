from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.app.api.databases import router as databases_router
from backend.app.api.predictions import router as predictions_router
from backend.app.config import get_settings
from backend.app.db.metadata import init_metadata_db

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def on_startup() -> None:
    init_metadata_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(ValueError)
def handle_value_error(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": str(exc),
                "message": "Request could not be processed.",
                "details": None,
            }
        },
    )


app.include_router(databases_router, prefix=settings.api_prefix)
app.include_router(predictions_router, prefix=settings.api_prefix)

