from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.training.config import load_finetune_config
from backend.app.training.fine_tune_runner import PlannerFineTuneRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planner fine-tuning trajectory loop.")
    parser.add_argument(
        "--config",
        default="config/training/relbench_finetune_config.example.json",
        help="Path to the fine-tuning configuration JSON file.",
    )
    args = parser.parse_args()
    config = load_finetune_config(args.config)
    result = PlannerFineTuneRunner(config).run()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
