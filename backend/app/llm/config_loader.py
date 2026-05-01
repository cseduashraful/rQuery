from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator

from backend.app.config import get_settings


class ProviderConfig(BaseModel):
    api_key_env: Optional[str] = None
    enabled: Optional[bool] = None
    backend: Optional[str] = None
    shared_base_model: Optional[str] = None
    task_planner_model: Optional[str] = None
    sql_explorer_model: Optional[str] = None
    final_predictor_model: Optional[str] = None
    summary_model: Optional[str] = None
    critic_model: Optional[str] = None
    model_path: Optional[str] = None
    model_paths: dict[str, str] = {}
    finetuned_model_root: Optional[str] = "./finetuned_llm"

    @model_validator(mode="after")
    def validate_model_resolution(self) -> "ProviderConfig":
        if self.shared_base_model:
            return self
        if self.task_planner_model and self.sql_explorer_model and self.final_predictor_model:
            return self
        raise ValueError(
            "Provider config must define either shared_base_model or explicit task_planner_model, "
            "sql_explorer_model, and final_predictor_model."
        )

    def model_for_role(self, role: str) -> str:
        if role == "task_planner":
            return self.task_planner_model or self.shared_base_model or ""
        if role == "sql_explorer":
            return self.sql_explorer_model or self.shared_base_model or self.task_planner_model or ""
        if role == "final_predictor":
            return self.final_predictor_model or self.shared_base_model or self.task_planner_model or ""
        if role == "summary":
            return self.summary_model or self.shared_base_model or self.task_planner_model or ""
        if role == "critic":
            return self.critic_model or self.shared_base_model or self.final_predictor_model or self.task_planner_model or ""
        raise KeyError(f"Unknown LLM role: {role}")

    def resolve_local_model_path(self, model_ref: str) -> str:
        finetuned_root = Path(self.finetuned_model_root or "./finetuned_llm")
        finetuned_candidate = finetuned_root / model_ref
        if finetuned_candidate.exists() and any(finetuned_candidate.iterdir()):
            return str(finetuned_candidate.resolve())
        if model_ref in self.model_paths:
            return self.model_paths[model_ref]
        if self.model_path:
            return self.model_path
        return model_ref

    def resolved_model_for_role(self, role: str, provider_name: str) -> str:
        model_ref = self.model_for_role(role)
        if provider_name == "local":
            return self.resolve_local_model_path(model_ref)
        return model_ref


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
