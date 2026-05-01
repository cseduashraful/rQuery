from __future__ import annotations

import random
from typing import Any

from backend.app.training.benchmark_loader import BenchmarkLoader
from backend.app.training.config import FinetuneConfig
from backend.app.training.engine import FineTuneExecutionEngine
from backend.app.training.planner_dataset_builder import RoleDatasetBuilder
from backend.app.training.planner_fine_tuner import PlannerFineTuner
from backend.app.training.trajectory_store import TrajectoryStore


class PlannerFineTuneRunner:
    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.loader = BenchmarkLoader(config)
        self.engine = FineTuneExecutionEngine()
        self.trajectory_store = TrajectoryStore(config.trainer.output_dir)
        self.dataset_builder = RoleDatasetBuilder(config.trainer.output_dir)
        self.fine_tuner = PlannerFineTuner(config.trainer)
        self.random = random.Random(config.loop.random_seed)

    def run(self) -> dict[str, Any]:
        episodes = []
        sufficient_count = 0
        matched_count = 0
        role_counts: dict[str, int] = {}
        for episode_index in range(self.config.loop.episodes):
            role = self.random.choices(
                self.config.loop.trainable_roles,
                weights=[self.config.loop.role_sampling_weights.get(item, 1.0) for item in self.config.loop.trainable_roles],
                k=1,
            )[0]
            sample = self.loader.random_sample()
            trajectory = self.engine.run_episode(sample, self.config.loop.max_refinement_rounds)
            trajectory["episode_index"] = episode_index
            trajectory["target_role"] = role
            trajectory["target_adapter"] = self.config.trainer.adapter_for_role(role).adapter_name
            self.trajectory_store.append(trajectory)
            self.dataset_builder.append_from_trajectory(trajectory, role)
            episodes.append(trajectory)
            role_counts[role] = role_counts.get(role, 0) + 1
            if trajectory["critic_report"]["prompt_sufficient"]:
                sufficient_count += 1
            if trajectory["final_evaluation"]["matched"]:
                matched_count += 1

        trainer_result = self.fine_tuner.maybe_run_hook(role_counts)
        state = {
            "episodes": len(episodes),
            "prompt_sufficiency_rate": sufficient_count / len(episodes) if episodes else 0.0,
            "ground_truth_match_rate": matched_count / len(episodes) if episodes else 0.0,
            "role_counts": role_counts,
            "trainer_result": trainer_result,
        }
        self.trajectory_store.write_state(state)
        return state
