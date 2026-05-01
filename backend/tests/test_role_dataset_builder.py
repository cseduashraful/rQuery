from __future__ import annotations

from pathlib import Path

from backend.app.training.planner_dataset_builder import RoleDatasetBuilder


def test_role_dataset_builder_writes_role_specific_example_files(tmp_path: Path) -> None:
    builder = RoleDatasetBuilder(str(tmp_path))
    trajectory = {
        "planner_input": {"question": "Q", "database_summary": "S", "memories": []},
        "initial_plan": {"needed_information": [], "assumptions": []},
        "critic_report": {"prompt_sufficient": False, "missing_information": ["payment"], "wrong_assumptions": []},
        "dataset_name": "demo",
        "task_name": "task",
        "entity_id": "1",
        "target_adapter": "planner-adapter",
        "final_evaluation": {"score": 1.0},
        "cutoff_time": "2024-01-01",
        "final_prompt": "prompt",
        "final_answer": {"summary": "ok"},
        "ground_truth": "paid",
        "evidence_packet": {"queries_run": [{"purpose": "p", "sql": "SELECT 1"}]},
    }
    builder.append_from_trajectory(trajectory, "task_planner")
    builder.append_from_trajectory(trajectory, "sql_explorer")
    builder.append_from_trajectory(trajectory, "final_predictor")
    builder.append_from_trajectory(trajectory, "critic")

    assert (tmp_path / "task_planner_training_examples.jsonl").exists()
    assert (tmp_path / "sql_explorer_training_examples.jsonl").exists()
    assert (tmp_path / "final_predictor_training_examples.jsonl").exists()
    assert (tmp_path / "critic_training_examples.jsonl").exists()
