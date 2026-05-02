from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.training.relbench_inference import RelBenchTerminalEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe RelBench test-split inference and report regression MAE."
    )
    parser.add_argument("--dataset", help="RelBench dataset name, e.g. rel-hm.")
    parser.add_argument("--task", help="RelBench task name, e.g. item-sales.")
    parser.add_argument(
        "--duckdb-path",
        help="Path to the generated DuckDB file. Defaults to output/relbench/<dataset>.duckdb.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/relbench_eval",
        help="Directory for JSONL predictions and summary output.",
    )
    parser.add_argument(
        "--question-template",
        help="Optional question template. Available keys include {entity_id}, {cutoff_time}, {target_col}, and row fields.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Number of test rows to process per batch.")
    parser.add_argument("--limit", type=int, help="Optional limit on the number of test rows to evaluate.")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to download/verify RelBench data through the official package cache.",
    )
    parser.add_argument(
        "--rebuild-duckdb",
        action="store_true",
        help="Rebuild the DuckDB file even if it already exists.",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List available RelBench datasets and exit.")
    parser.add_argument("--list-tasks", metavar="DATASET", help="List available tasks for a dataset and exit.")

    args = parser.parse_args()
    evaluator = RelBenchTerminalEvaluator()

    if args.list_datasets:
        print(json.dumps({"datasets": evaluator.list_datasets()}, indent=2))
        return
    if args.list_tasks:
        print(json.dumps({"dataset": args.list_tasks, "tasks": evaluator.list_tasks(args.list_tasks)}, indent=2))
        return

    if not args.dataset or not args.task:
        parser.error("--dataset and --task are required unless you use --list-datasets or --list-tasks.")

    duckdb_path = args.duckdb_path or f"output/relbench/{args.dataset}.duckdb"
    summary = evaluator.evaluate_test_split(
        dataset_name=args.dataset,
        task_name=args.task,
        duckdb_path=duckdb_path,
        output_dir=args.output_dir,
        question_template=args.question_template,
        batch_size=args.batch_size,
        limit=args.limit,
        download=args.download,
        rebuild_duckdb=args.rebuild_duckdb,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
