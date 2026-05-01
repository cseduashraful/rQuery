from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Optional


class RoleDatasetBuilder:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_path_by_role = {
            "task_planner": self.output_dir / "task_planner_training_examples.jsonl",
            "sql_explorer": self.output_dir / "sql_explorer_training_examples.jsonl",
            "final_predictor": self.output_dir / "final_predictor_training_examples.jsonl",
            "critic": self.output_dir / "critic_training_examples.jsonl",
        }

    def append_from_trajectory(self, trajectory: dict[str, Any], role: str) -> None:
        example = self._build_example(trajectory, role)
        if not example:
            return
        path = self.examples_path_by_role[role]
        with path.open("a") as handle:
            handle.write(json.dumps(example, default=str) + "\n")

    def _build_example(self, trajectory: dict[str, Any], role: str) -> Optional[dict[str, Any]]:
        if role == "task_planner":
            return self._build_planner_example(trajectory)
        if role == "sql_explorer":
            return self._build_explorer_example(trajectory)
        if role == "final_predictor":
            return self._build_predictor_example(trajectory)
        if role == "critic":
            return self._build_critic_example(trajectory)
        raise ValueError(f"Unsupported training role: {role}")

    def _build_planner_example(self, trajectory: dict[str, Any]) -> Optional[dict[str, Any]]:
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
                "target_role": "task_planner",
                "target_adapter": trajectory["target_adapter"],
                "dataset": trajectory["dataset_name"],
                "task": trajectory["task_name"],
                "entity_id": trajectory["entity_id"],
                "ground_truth_score": trajectory["final_evaluation"]["score"],
            },
        }

    def _build_explorer_example(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        user_payload = {
            "task": trajectory["initial_plan"],
            "schema_summary": trajectory["planner_input"]["database_summary"],
            "available_tables": trajectory["planner_input"]["memories"],
            "cutoff_time": trajectory["cutoff_time"],
        }
        query_targets = [
            {"purpose": item["purpose"], "sql": item["sql"]}
            for item in trajectory["evidence_packet"].get("queries_run", [])
        ]
        return {
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
                {"role": "assistant", "content": json.dumps({"done": False, "queries": query_targets}, default=str)},
            ],
            "metadata": {
                "target_role": "sql_explorer",
                "target_adapter": trajectory["target_adapter"],
                "dataset": trajectory["dataset_name"],
                "task": trajectory["task_name"],
                "entity_id": trajectory["entity_id"],
            },
        }

    def _build_predictor_example(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": trajectory["final_prompt"]},
                {"role": "assistant", "content": json.dumps(trajectory["final_answer"], default=str)},
            ],
            "metadata": {
                "target_role": "final_predictor",
                "target_adapter": trajectory["target_adapter"],
                "dataset": trajectory["dataset_name"],
                "task": trajectory["task_name"],
                "entity_id": trajectory["entity_id"],
                "ground_truth_score": trajectory["final_evaluation"]["score"],
            },
        }

    def _build_critic_example(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        user_payload = {
            "final_prompt": trajectory["final_prompt"],
            "final_answer": trajectory["final_answer"],
            "ground_truth": trajectory["ground_truth"],
        }
        return {
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
                {"role": "assistant", "content": json.dumps(trajectory["critic_report"], default=str)},
            ],
            "metadata": {
                "target_role": "critic",
                "target_adapter": trajectory["target_adapter"],
                "dataset": trajectory["dataset_name"],
                "task": trajectory["task_name"],
                "entity_id": trajectory["entity_id"],
            },
        }
