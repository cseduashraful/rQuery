from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from backend.app.training.relbench_inference import RelBenchDatabaseMaterializer
from backend.app.training.relbench_inference import RelBenchTerminalEvaluator


class FakeTable:
    def __init__(self, df: pd.DataFrame, pkey_col: str | None = None, time_col: str | None = None) -> None:
        self.df = df
        self.pkey_col = pkey_col
        self.time_col = time_col
        self.fkey_col_to_pkey_table: dict[str, str] = {}


class FakeDatabase:
    def __init__(self) -> None:
        self.table_dict = {
            "customers": FakeTable(
                pd.DataFrame(
                    [
                        {"customer_id": 1, "created_at": "2024-01-01", "segment": "A"},
                        {"customer_id": 2, "created_at": "2024-01-02", "segment": "B"},
                    ]
                ),
                pkey_col="customer_id",
                time_col="created_at",
            )
        }


def test_relbench_materializer_writes_duckdb(tmp_path: Path) -> None:
    materializer = RelBenchDatabaseMaterializer()
    db_path = materializer.materialize(FakeDatabase(), str(tmp_path / "fake.duckdb"), rebuild=True)

    conn = duckdb.connect(db_path, read_only=True)
    try:
        row_count = conn.execute('SELECT COUNT(*) FROM "customers"').fetchone()[0]
    finally:
        conn.close()

    assert row_count == 2
    metadata = json.loads((tmp_path / "fake.metadata.json").read_text())
    assert metadata["customers"]["primary_key"] == "customer_id"


def test_relbench_question_template_defaults() -> None:
    evaluator = RelBenchTerminalEvaluator()
    question = evaluator._render_question(
        template=None,
        entity_id="42",
        cutoff_time="2024-02-01 00:00:00",
        target_col="sales",
        entity_col="item_id",
        row_payload={},
    )
    assert "2024-02-01" in question
    assert "sales" in question
    assert "item_id=42" in question


def test_relbench_numeric_coercion() -> None:
    evaluator = RelBenchTerminalEvaluator()
    assert evaluator._coerce_float("3.5") == 3.5
    assert evaluator._coerce_float(7) == 7.0
    assert evaluator._coerce_float("not-a-number") is None


def test_relbench_safe_format_dict_preserves_unknown_keys() -> None:
    evaluator = RelBenchTerminalEvaluator()
    question = evaluator._render_question(
        template="Predict {target_col} for {entity_id} using {unknown_field}.",
        entity_id="9",
        cutoff_time="2024-02-01",
        target_col="sales",
        entity_col="item_id",
        row_payload={"known": "value"},
    )
    assert "{unknown_field}" in question
