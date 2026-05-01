from __future__ import annotations

from pathlib import Path

from backend.app.training.adapter_trainer import AdapterTrainer
from backend.app.training.config import PlannerTrainerConfig


def test_adapter_trainer_builds_role_output_path(tmp_path: Path) -> None:
    trainer = AdapterTrainer(
        PlannerTrainerConfig(
            output_dir=str(tmp_path / "output"),
            base_model_name="8b",
            finetuned_model_root=str(tmp_path / "finetuned_llm"),
        )
    )
    plan = trainer.build_plan("task_planner")

    assert plan.base_model_ref == "8b"
    assert plan.dataset_path.endswith("task_planner_training_examples.jsonl")
    assert plan.adapter_output_path.endswith("finetuned_llm/8b/task_planner")

