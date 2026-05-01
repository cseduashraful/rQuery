from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.training.config import FinetuneConfig, FinetuneDatasetConfig, FinetuneTaskConfig


@dataclass
class BenchmarkSample:
    dataset: FinetuneDatasetConfig
    task: FinetuneTaskConfig
    entity: dict[str, Any]
    question: str
    entity_id: str
    ground_truth: Any
    cutoff_time: str


class BenchmarkLoader:
    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.random = random.Random(config.loop.random_seed)

    def random_sample(self) -> BenchmarkSample:
        dataset = self.random.choice(self.config.datasets)
        task = self.random.choice(dataset.tasks)
        rows = self._load_rows(task.train_split_path)
        entity = self.random.choice(rows)
        question = self._render_question(task, entity)
        return BenchmarkSample(
            dataset=dataset,
            task=task,
            entity=entity,
            question=question,
            entity_id=str(entity[task.entity_id_column]),
            ground_truth=entity[task.ground_truth_column],
            cutoff_time=str(entity[task.cutoff_time_column]),
        )

    def _render_question(self, task: FinetuneTaskConfig, entity: dict[str, Any]) -> str:
        payload = dict(entity)
        payload.setdefault("entity_id", entity.get(task.entity_id_column))
        return task.question_template.format(**payload)

    def _load_rows(self, path: str) -> list[dict[str, Any]]:
        file_path = Path(path)
        if file_path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in file_path.read_text().splitlines() if line.strip()]
        if file_path.suffix.lower() == ".json":
            return json.loads(file_path.read_text())
        if file_path.suffix.lower() == ".csv":
            with file_path.open(newline="") as handle:
                return list(csv.DictReader(handle))
        raise ValueError(f"Unsupported training split format: {path}")

