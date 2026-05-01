from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from backend.app.training.adapter_trainer import AdapterTrainer
from backend.app.training.config import PlannerTrainerConfig


class PlannerFineTuner:
    def __init__(self, trainer_config: PlannerTrainerConfig) -> None:
        self.trainer_config = trainer_config
        self.output_dir = Path(trainer_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_trainer = AdapterTrainer(trainer_config)

    def maybe_run_hook(self, role_counts: dict[str, int]) -> dict[str, Any]:
        active_roles = [role for role, count in role_counts.items() if count > 0]
        active_adapters = {
            role: self.trainer_config.adapter_for_role(role).adapter_name for role in active_roles
        }
        if self.trainer_config.mode == "peft_lora":
            training_results = []
            for role in active_roles:
                training_results.append(self.adapter_trainer.train_role(role))
            return {
                "trainer_mode": self.trainer_config.mode,
                "hook_executed": False,
                "base_model_name": self.trainer_config.base_model_name,
                "active_roles": active_roles,
                "active_adapters": active_adapters,
                "training_results": training_results,
            }
        if not self.trainer_config.hook_command:
            return {
                "trainer_mode": self.trainer_config.mode,
                "hook_executed": False,
                "base_model_name": self.trainer_config.base_model_name,
                "active_roles": active_roles,
                "active_adapters": active_adapters,
            }
        result = subprocess.run(
            shlex.split(self.trainer_config.hook_command),
            cwd=str(self.output_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "trainer_mode": self.trainer_config.mode,
            "hook_executed": True,
            "base_model_name": self.trainer_config.base_model_name,
            "active_roles": active_roles,
            "active_adapters": active_adapters,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
