from __future__ import annotations

from pathlib import Path

from backend.app.llm.local_runtime import LocalRuntimeManager


class _FakeLoadedModel:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path


def test_local_runtime_manager_caches_one_loaded_model_per_path(monkeypatch) -> None:
    LocalRuntimeManager.clear_cache()
    load_calls: list[str] = []

    def fake_load(model_path: str):
        load_calls.append(model_path)
        return _FakeLoadedModel(model_path)

    monkeypatch.setattr(LocalRuntimeManager, "_load_model", fake_load)

    first = LocalRuntimeManager.get_model("/tmp/model-a")
    second = LocalRuntimeManager.get_model("/tmp/model-a")
    third = LocalRuntimeManager.get_model("/tmp/model-b")

    assert first is second
    assert first is not third
    assert load_calls == [str(Path("/tmp/model-a").resolve()), str(Path("/tmp/model-b").resolve())]


class _FakeAdapterModel:
    def __init__(self) -> None:
        self.loaded = []
        self.current = None

    def load_adapter(self, adapter_path: str, adapter_name: str) -> None:
        self.loaded.append((adapter_path, adapter_name))

    def set_adapter(self, adapter_name: str) -> None:
        self.current = adapter_name


def test_loaded_local_model_reuses_loaded_adapter() -> None:
    from backend.app.llm.local_runtime import LoadedLocalModel

    model = LoadedLocalModel(model_path="/base/model", tokenizer=None, model=_FakeAdapterModel(), loaded_adapters={})
    model.loaded_adapters["/adapters/task_planner"] = "task_planner"
    adapter_name = model.ensure_adapter("/adapters/task_planner", "task_planner")
    assert adapter_name == "task_planner"
    assert model.model.loaded == []
