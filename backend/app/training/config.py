from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class FinetuneTaskConfig(BaseModel):
    name: str
    question_template: str
    train_split_path: str
    entity_id_column: str
    entity_lookup_column: Optional[str] = None
    ground_truth_column: str
    cutoff_time_column: str
    candidate_tables: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    question_variables: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_no_ground_truth_leakage(self) -> "FinetuneTaskConfig":
        forbidden_token = "{" + self.ground_truth_column + "}"
        if forbidden_token in self.question_template:
            raise ValueError(
                f"Question template for task '{self.name}' leaks the ground truth via {forbidden_token}."
            )
        return self


class FinetuneDatasetConfig(BaseModel):
    name: str
    database_path: str
    display_name: str
    tasks: list[FinetuneTaskConfig]


class FinetuneLoopConfig(BaseModel):
    random_seed: int = 7
    episodes: int = 10
    max_refinement_rounds: int = 2
    trainer_user_id: str = "trainer_user"
    trainable_roles: list[str] = Field(
        default_factory=lambda: ["task_planner", "sql_explorer", "final_predictor", "critic"]
    )
    role_sampling_weights: dict[str, float] = Field(default_factory=dict)

    def choose_role(self) -> str:
        if not self.trainable_roles:
            raise ValueError("trainable_roles cannot be empty.")
        weights = [self.role_sampling_weights.get(role, 1.0) for role in self.trainable_roles]
        rng = random.Random(self.random_seed)
        return rng.choices(self.trainable_roles, weights=weights, k=1)[0]


class RoleAdapterConfig(BaseModel):
    adapter_name: str
    adapter_path: Optional[str] = None
    enabled: bool = True


class PlannerTrainerConfig(BaseModel):
    mode: str = "jsonl_export"
    output_dir: str = "./output/training"
    min_examples_before_emit: int = 1
    hook_command: Optional[str] = None
    base_model_name: str = "shared-base-model"
    adapters: dict[str, RoleAdapterConfig] = Field(
        default_factory=lambda: {
            "task_planner": RoleAdapterConfig(adapter_name="task-planner-adapter"),
            "sql_explorer": RoleAdapterConfig(adapter_name="sql-explorer-adapter"),
            "final_predictor": RoleAdapterConfig(adapter_name="final-predictor-adapter"),
            "critic": RoleAdapterConfig(adapter_name="critic-adapter"),
        }
    )

    def adapter_for_role(self, role: str) -> RoleAdapterConfig:
        if role not in self.adapters:
            raise ValueError(f"No adapter configured for role '{role}'.")
        return self.adapters[role]


class FinetuneConfig(BaseModel):
    datasets: list[FinetuneDatasetConfig]
    loop: FinetuneLoopConfig = Field(default_factory=FinetuneLoopConfig)
    trainer: PlannerTrainerConfig = Field(default_factory=PlannerTrainerConfig)


def load_finetune_config(path: str) -> FinetuneConfig:
    payload = json.loads(Path(path).read_text())
    return FinetuneConfig.model_validate(payload)
