from __future__ import annotations

import json
from pathlib import Path

from backend.app.training.config import FinetuneConfig
from backend.app.training.fine_tune_runner import PlannerFineTuneRunner


def test_finetune_loop_generates_outputs(tmp_path: Path) -> None:
    db_path = Path("sample_data/ecommerce.duckdb").resolve()
    split_path = Path("backend/tests/fixtures/finetune_train_split.jsonl").resolve()
    output_dir = tmp_path / "training_output"
    config = FinetuneConfig.model_validate(
        {
            "datasets": [
                {
                    "name": "ecommerce_fixture",
                    "database_path": str(db_path),
                    "display_name": "Ecommerce Fixture",
                    "tasks": [
                        {
                            "name": "payment_status_prediction",
                            "question_template": "Based only on information available before {cutoff_time}, what is the likely payment outcome for customer {customer_id}?",
                            "train_split_path": str(split_path),
                            "entity_id_column": "customer_id",
                            "entity_lookup_column": "customer_id",
                            "ground_truth_column": "ground_truth",
                            "cutoff_time_column": "cutoff_time",
                            "candidate_tables": ["payments", "customers"],
                            "expected_evidence": ["payment", "customer"],
                        }
                    ],
                }
            ],
            "loop": {"episodes": 2, "max_refinement_rounds": 1, "random_seed": 3},
            "trainer": {"output_dir": str(output_dir)},
        }
    )

    result = PlannerFineTuneRunner(config).run()

    assert result["episodes"] == 2
    assert (output_dir / "trajectories.jsonl").exists()
    assert (output_dir / "planner_training_examples.jsonl").exists()
    state = json.loads((output_dir / "training_state.json").read_text())
    assert "prompt_sufficiency_rate" in state
    assert state["ground_truth_match_rate"] >= 0.5
