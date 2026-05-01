from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from backend.app.training.config import PlannerTrainerConfig


class PlannerFineTuner:
    def __init__(self, trainer_config: PlannerTrainerConfig) -> None:
        self.trainer_config = trainer_config
        self.output_dir = Path(trainer_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def maybe_run_hook(self) -> dict:
        if not self.trainer_config.hook_command:
            return {"trainer_mode": self.trainer_config.mode, "hook_executed": False}
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
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

