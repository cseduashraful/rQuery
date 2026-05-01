from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from typing import Optional


class LocalLLMImportError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = stripped.find("{", start + 1)
    raise ValueError("Local model response did not contain valid JSON.")


@dataclass
class LoadedLocalModel:
    model_path: str
    tokenizer: Any
    model: Any
    loaded_adapters: dict[str, str] | None = None

    def ensure_adapter(self, adapter_path: Optional[str], role: str) -> Optional[str]:
        if not adapter_path:
            return None
        if self.loaded_adapters is None:
            self.loaded_adapters = {}
        if adapter_path in self.loaded_adapters:
            return self.loaded_adapters[adapter_path]

        adapter_name = role
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise LocalLLMImportError(
                "Local adapter inference requires peft. Use environment.local-inference.yml or "
                "environment.finetune.yml."
            ) from exc

        if not self.loaded_adapters:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=adapter_name)
        else:
            if not hasattr(self.model, "load_adapter"):
                raise LocalLLMImportError(
                    "Loaded local model does not support dynamic adapter loading."
                )
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
        self.loaded_adapters[adapter_path] = adapter_name
        return adapter_name

    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int = 768,
        adapter_path: Optional[str] = None,
        role: str = "",
    ) -> dict[str, Any]:
        adapter_name = self.ensure_adapter(adapter_path, role) if adapter_path else None
        if adapter_name and hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            model_inputs = {key: value.to(self.model.device) for key, value in model_inputs.items()}
        output = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _extract_json_object(text)


class LocalRuntimeManager:
    _instances: dict[str, LoadedLocalModel] = {}
    _lock = Lock()

    @classmethod
    def get_model(cls, model_path: str) -> LoadedLocalModel:
        resolved = str(Path(model_path).resolve())
        with cls._lock:
            if resolved not in cls._instances:
                cls._instances[resolved] = cls._load_model(resolved)
            return cls._instances[resolved]

    @classmethod
    def _load_model(cls, model_path: str) -> LoadedLocalModel:
        try:
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise LocalLLMImportError(
                "Local LLM execution requires transformers and torch. "
                "Use environment.local-inference.yml or environment.finetune.yml."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        return LoadedLocalModel(model_path=model_path, tokenizer=tokenizer, model=model, loaded_adapters={})

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instances.clear()
