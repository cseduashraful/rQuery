from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.training.adapter_trainer import AdapterTrainer
from backend.app.training.config import load_finetune_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adapter fine-tuning for one or more roles.")
    parser.add_argument(
        "--config",
        default="config/training/relbench_finetune_config.example.json",
        help="Path to the fine-tuning configuration JSON file.",
    )
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        help="Role(s) to train. May be provided multiple times. Defaults to configured trainable_roles.",
    )
    args = parser.parse_args()

    config = load_finetune_config(args.config)
    trainer = AdapterTrainer(config.trainer)
    roles = args.roles or config.loop.trainable_roles
    results = [trainer.train_role(role) for role in roles]
    print(json.dumps({"roles": roles, "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
