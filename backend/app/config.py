from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DuckDB Predictive Agent"
    environment: str = "development"
    api_prefix: str = "/api"
    default_user_id: str = "dev_user"
    metadata_database_url: str = "sqlite:///./storage/metadata.sqlite3"
    storage_root: Path = Path("./storage")
    upload_root: Path = Path("./storage/uploads")
    llm_config_path: Path = Path("./config/llm_config.json")
    max_agent_steps: int = 5
    max_queries_per_prediction: int = 10
    max_rows_per_query: int = 1000
    max_rows_to_llm: int = 100
    query_timeout_seconds: int = 10
    enable_llm_calls: bool = False
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    settings.upload_root.mkdir(parents=True, exist_ok=True)
    return settings
