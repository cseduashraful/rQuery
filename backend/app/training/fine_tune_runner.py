from __future__ import annotations

from typing import Any

from backend.app.training.benchmark_loader import BenchmarkLoader
from backend.app.training.config import FinetuneConfig
from backend.app.training.engine import FineTuneExecutionEngine
from backend.app.training.planner_dataset_builder import PlannerDatasetBuilder
from backend.app.training.planner_fine_tuner import PlannerFineTuner
from backend.app.training.trajectory_store import TrajectoryStore


class PlannerFineTuneRunner:
    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.loader = BenchmarkLoader(config)
        self.engine = FineTuneExecutionEngine()
        self.trajectory_store = TrajectoryStore(config.trainer.output_dir)
        self.dataset_builder = PlannerDatasetBuilder(config.trainer.output_dir)
        self.fine_tuner = PlannerFineTuner(config.trainer)

    def run(self) -> dict[str, Any]:
        episodes = []
        sufficient_count = 0
        matched_count = 0
        for episode_index in range(self.config.loop.episodes):
            sample = self.loader.random_sample()
            trajectory = self.engine.run_episode(sample, self.config.loop.max_refinement_rounds)
            trajectory["episode_index"] = episode_index
            self.trajectory_store.append(trajectory)
            self.dataset_builder.append_from_trajectory(trajectory)
            episodes.append(trajectory)
            if trajectory["critic_report"]["prompt_sufficient"]:
                sufficient_count += 1
            if trajectory["final_evaluation"]["matched"]:
                matched_count += 1

        trainer_result = self.fine_tuner.maybe_run_hook()
        state = {
            "episodes": len(episodes),
            "prompt_sufficiency_rate": sufficient_count / len(episodes) if episodes else 0.0,
            "ground_truth_match_rate": matched_count / len(episodes) if episodes else 0.0,
            "trainer_result": trainer_result,
        }
        self.trajectory_store.write_state(state)
        return state

