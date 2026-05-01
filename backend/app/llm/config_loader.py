from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from backend.app.config import get_settings


class ProviderConfig(BaseModel):
    api_key_env: Optional[str] = None
    enabled: Optional[bool] = None
    backend: Optional[str] = None
    task_planner_model: str
    sql_explorer_model: str
    final_predictor_model: str
    summary_model: Optional[str] = None
    model_path: Optional[str] = None


class LLMConfig(BaseModel):
    default_provider: str
    providers: dict[str, ProviderConfig]


def load_llm_config() -> LLMConfig:
    settings = get_settings()
    path = Path(settings.llm_config_path)
    if not path.exists():
        example = Path("./config/llm_config.example.json")
        path = example
    return LLMConfig.model_validate(json.loads(path.read_text()))
