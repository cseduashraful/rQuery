from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Optional


class PlannerDatasetBuilder:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_path = self.output_dir / "planner_training_examples.jsonl"

    def append_from_trajectory(self, trajectory: dict[str, Any]) -> None:
        example = self._build_example(trajectory)
        if not example:
            return
        with self.examples_path.open("a") as handle:
            handle.write(json.dumps(example, default=str) + "\n")

    def _build_example(self, trajectory: dict[str, Any]) -> Optional[dict[str, Any]]:
        planner_input = trajectory["planner_input"]
        initial_plan = trajectory["initial_plan"]
        critic = trajectory["critic_report"]
        improved_plan = dict(initial_plan)

        if critic.get("missing_information"):
            existing = set(improved_plan.get("needed_information", []))
            for item in critic["missing_information"]:
                if item not in existing:
                    improved_plan.setdefault("needed_information", []).append(item)
        if critic.get("wrong_assumptions"):
            improved_plan["assumptions"] = [
                assumption
                for assumption in improved_plan.get("assumptions", [])
                if assumption not in critic["wrong_assumptions"]
            ]
        improved_plan["critic_guided"] = True
        improved_plan["prompt_sufficient"] = critic.get("prompt_sufficient", False)

        return {
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {
                    "role": "user",
                    "content": json.dumps(planner_input, default=str),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(improved_plan, default=str),
                },
            ],
            "metadata": {
                "dataset": trajectory["dataset_name"],
                "task": trajectory["task_name"],
                "entity_id": trajectory["entity_id"],
                "ground_truth_score": trajectory["final_evaluation"]["score"],
            },
        }
