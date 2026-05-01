from __future__ import annotations

from pathlib import Path

from backend.app.llm.config_loader import ProviderConfig


def test_local_model_resolution_prefers_merged_finetuned_copy(tmp_path: Path) -> None:
    finetuned_root = tmp_path / "finetuned_llm"
    model_dir = finetuned_root / "8b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    provider = ProviderConfig.model_validate(
        {
            "shared_base_model": "8b",
            "finetuned_model_root": str(finetuned_root),
            "model_paths": {"8b": "/base/models/8b"},
        }
    )

    assert provider.resolved_model_for_role("sql_explorer", "local") == str(model_dir.resolve())


def test_local_model_resolution_prefers_role_adapter_when_present(tmp_path: Path) -> None:
    finetuned_root = tmp_path / "finetuned_llm"
    adapter_dir = finetuned_root / "8b" / "task_planner"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}")

    provider = ProviderConfig.model_validate(
        {
            "shared_base_model": "8b",
            "task_planner_model": "8b",
            "finetuned_model_root": str(finetuned_root),
            "model_paths": {"8b": "/base/models/8b"},
        }
    )

    assert provider.resolved_model_for_role("task_planner", "local") == "/base/models/8b"
    assert provider.resolved_adapter_for_role("task_planner", "local") == str(adapter_dir.resolve())


def test_local_model_resolution_falls_back_to_base_path(tmp_path: Path) -> None:
    finetuned_root = tmp_path / "finetuned_llm"
    finetuned_root.mkdir(parents=True)

    provider = ProviderConfig.model_validate(
        {
            "shared_base_model": "3b",
            "finetuned_model_root": str(finetuned_root),
            "model_paths": {"3b": "/base/models/3b"},
        }
    )

    assert provider.resolved_model_for_role("final_predictor", "local") == "/base/models/3b"
