from __future__ import annotations

from backend.app.llm.config_loader import ProviderConfig


def test_provider_config_supports_shared_base_plus_planner_override() -> None:
    provider = ProviderConfig.model_validate(
        {
            "shared_base_model": "base-model",
            "task_planner_model": "planner-ft-model",
        }
    )

    assert provider.model_for_role("task_planner") == "planner-ft-model"
    assert provider.model_for_role("sql_explorer") == "base-model"
    assert provider.model_for_role("final_predictor") == "base-model"
    assert provider.model_for_role("summary") == "base-model"
    assert provider.model_for_role("critic") == "base-model"


def test_provider_config_supports_explicit_role_models_without_shared_base() -> None:
    provider = ProviderConfig.model_validate(
        {
            "task_planner_model": "planner-model",
            "sql_explorer_model": "explorer-model",
            "final_predictor_model": "predictor-model",
            "summary_model": "summary-model",
            "critic_model": "critic-model",
        }
    )

    assert provider.model_for_role("task_planner") == "planner-model"
    assert provider.model_for_role("sql_explorer") == "explorer-model"
    assert provider.model_for_role("final_predictor") == "predictor-model"
    assert provider.model_for_role("summary") == "summary-model"
    assert provider.model_for_role("critic") == "critic-model"
