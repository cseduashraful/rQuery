from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrajectoryStore:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.output_dir / "trajectories.jsonl"
        self.state_path = self.output_dir / "training_state.json"

    def append(self, trajectory: dict[str, Any]) -> None:
        with self.trajectory_path.open("a") as handle:
            handle.write(json.dumps(trajectory, default=str) + "\n")

    def write_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

