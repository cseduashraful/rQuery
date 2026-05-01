from __future__ import annotations

import json
from typing import Any
from typing import Optional

from openai import OpenAI

from backend.app.config import get_settings
from backend.app.llm.config_loader import ProviderConfig, load_llm_config
from backend.app.llm.local_runtime import LocalRuntimeManager


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.config = load_llm_config()

    def provider(self, name: Optional[str] = None) -> ProviderConfig:
        provider_name = name or self.config.default_provider
        return self.config.providers[provider_name]

    def resolved_model_for_role(self, task_type: str, provider_name: Optional[str] = None) -> str:
        provider_key = provider_name or self.config.default_provider
        provider = self.provider(provider_key)
        return provider.resolved_model_for_role(task_type, provider_key)

    def resolved_adapter_for_role(self, task_type: str, provider_name: Optional[str] = None) -> Optional[str]:
        provider_key = provider_name or self.config.default_provider
        provider = self.provider(provider_key)
        return provider.resolved_adapter_for_role(task_type, provider_key)

    def generate_json(
        self,
        task_type: str,
        prompt: str,
        schema: dict[str, Any],
        provider_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self.settings.enable_llm_calls:
            raise RuntimeError("LLM calls are disabled. Enable them via configuration.")

        provider_key = provider_name or self.config.default_provider
        model_name = self.resolved_model_for_role(task_type, provider_key)

        if provider_key == "local":
            runtime = LocalRuntimeManager.get_model(model_name)
            adapter_path = self.resolved_adapter_for_role(task_type, provider_key)
            local_prompt = self._build_local_json_prompt(task_type, prompt, schema)
            return runtime.generate_json(local_prompt, adapter_path=adapter_path, role=task_type)

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.responses.create(
            model=model_name,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"{task_type}_response",
                    "schema": schema,
                }
            },
        )
        text = response.output_text
        return json.loads(text)

    def _build_local_json_prompt(self, task_type: str, prompt: str, schema: dict[str, Any]) -> str:
        return (
            f"You are handling the role: {task_type}.\n"
            "Return one valid JSON object only.\n"
            "Do not add prose, markdown, or explanations.\n\n"
            f"JSON SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
            f"TASK INPUT:\n{prompt}\n"
        )
