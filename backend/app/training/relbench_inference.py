from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from typing import Optional

import duckdb

from backend.app.agent.predictor import RegressionPredictor
from backend.app.config import get_settings
from backend.app.training.engine import DatasetRuntimeContext
from backend.app.training.engine import FineTuneExecutionEngine


@dataclass
class RuntimeDatasetRef:
    name: str
    database_path: str


@dataclass
class RuntimeTaskRef:
    name: str
    entity_id_column: str
    entity_lookup_column: Optional[str]
    ground_truth_column: str
    candidate_tables: list[str]
    expected_evidence: list[str]


@dataclass
class RuntimeSample:
    dataset: RuntimeDatasetRef
    task: RuntimeTaskRef
    question: str
    entity_id: str
    ground_truth: Any
    cutoff_time: str


@dataclass
class RelBenchTaskSpec:
    dataset_name: str
    task_name: str
    entity_col: str
    entity_table: str
    time_col: str
    target_col: str
    task_type: str
    candidate_tables: list[str]
    expected_evidence: list[str]


class RelBenchImportError(RuntimeError):
    pass


def _require_relbench():
    try:
        from relbench.datasets import get_dataset
        from relbench.datasets import get_dataset_names
        from relbench.tasks import get_task
        from relbench.tasks import get_task_names
    except ImportError as exc:
        raise RelBenchImportError(
            "RelBench support requires the `relbench` package. "
            "Use environment.local-inference.yml or environment.finetune.yml."
        ) from exc
    return get_dataset, get_dataset_names, get_task, get_task_names


class RelBenchDatabaseMaterializer:
    def materialize(self, database: Any, duckdb_path: str, rebuild: bool = False) -> str:
        output_path = Path(duckdb_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if rebuild and output_path.exists():
            output_path.unlink()
        if output_path.exists():
            return str(output_path)

        conn = duckdb.connect(str(output_path))
        metadata: dict[str, Any] = {}
        try:
            for table_name, table in database.table_dict.items():
                df = table.df.copy()
                conn.register("_relbench_df", df)
                quoted_name = self._quote_ident(table_name)
                conn.execute(f"CREATE OR REPLACE TABLE {quoted_name} AS SELECT * FROM _relbench_df")
                conn.unregister("_relbench_df")
                metadata[table_name] = {
                    "primary_key": table.pkey_col,
                    "time_col": table.time_col,
                    "foreign_keys": table.fkey_col_to_pkey_table,
                    "row_count": int(len(df.index)),
                }
        finally:
            conn.close()

        output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        return str(output_path)

    def _quote_ident(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'


class RelBenchTerminalEvaluator:
    def __init__(self) -> None:
        self.engine = FineTuneExecutionEngine()
        self.predictor = RegressionPredictor()
        self.materializer = RelBenchDatabaseMaterializer()
        self.settings = get_settings()

    def list_datasets(self) -> list[str]:
        _get_dataset, get_dataset_names, _get_task, _get_task_names = _require_relbench()
        return list(get_dataset_names())

    def list_tasks(self, dataset_name: str) -> list[str]:
        _get_dataset, _get_dataset_names, _get_task, get_task_names = _require_relbench()
        return list(get_task_names(dataset_name))

    def evaluate_test_split(
        self,
        dataset_name: str,
        task_name: str,
        duckdb_path: str,
        output_dir: str,
        question_template: Optional[str] = None,
        batch_size: int = 16,
        limit: Optional[int] = None,
        download: bool = True,
        rebuild_duckdb: bool = False,
    ) -> dict[str, Any]:
        get_dataset, _get_dataset_names, get_task, _get_task_names = _require_relbench()
        dataset = get_dataset(dataset_name, download=download)
        task = get_task(dataset_name, task_name, download=download)
        spec = self._build_task_spec(dataset_name, task_name, dataset, task)
        if spec.task_type != "regression":
            raise ValueError(
                f"Task {dataset_name}/{task_name} is '{spec.task_type}', not regression. "
                "This terminal evaluator currently reports MAE for regression tasks only."
            )
        if question_template and ("{" + spec.target_col + "}") in question_template:
            raise ValueError(
                f"Question template leaks the hidden target via {{{spec.target_col}}}. "
                "Use only masked test-row fields and the safe built-in variables."
            )

        database_path = self.materializer.materialize(
            dataset.get_db(upto_test_timestamp=False),
            duckdb_path=duckdb_path,
            rebuild=rebuild_duckdb,
        )
        context = self.engine.ensure_context(database_path)
        masked_test = task.get_table("test").df.reset_index(drop=True)
        full_test = task.get_table("test", mask_input_cols=False).df.reset_index(drop=True)
        train_table = task.get_table("train", mask_input_cols=False).df
        train_mean = float(train_table[spec.target_col].mean())

        total_rows = len(masked_test.index)
        if limit is not None:
            total_rows = min(total_rows, limit)

        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_path = output_path / f"{dataset_name}__{task_name}__predictions.jsonl"

        predictions: list[float] = []
        targets: list[float] = []
        leakage_failures = 0

        with predictions_path.open("w") as handle:
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                for row_index in range(batch_start, batch_end):
                    masked_row = masked_test.iloc[row_index].to_dict()
                    full_row = full_test.iloc[row_index].to_dict()
                    cutoff_time = str(masked_row[spec.time_col])
                    entity_id = str(masked_row[spec.entity_col])
                    question = self._render_question(
                        question_template,
                        entity_id=entity_id,
                        cutoff_time=cutoff_time,
                        target_col=spec.target_col,
                        entity_col=spec.entity_col,
                        row_payload=masked_row,
                    )
                    sample = RuntimeSample(
                        dataset=RuntimeDatasetRef(name=dataset_name, database_path=database_path),
                        task=RuntimeTaskRef(
                            name=task_name,
                            entity_id_column=spec.entity_col,
                            entity_lookup_column=spec.entity_col,
                            ground_truth_column=spec.target_col,
                            candidate_tables=spec.candidate_tables,
                            expected_evidence=spec.expected_evidence,
                        ),
                        question=question,
                        entity_id=entity_id,
                        ground_truth=full_row[spec.target_col],
                        cutoff_time=cutoff_time,
                    )
                    prediction_record = self._run_single_prediction(
                        sample=sample,
                        task_spec=spec,
                        context=context,
                        train_mean=train_mean,
                    )
                    if not prediction_record["leakage_audit"]["passed"]:
                        leakage_failures += 1
                    predictions.append(prediction_record["predicted_value"])
                    targets.append(float(prediction_record["ground_truth"]))
                    handle.write(json.dumps(prediction_record, default=str) + "\n")

        mae = mean(abs(pred - truth) for pred, truth in zip(predictions, targets)) if predictions else 0.0
        summary = {
            "dataset_name": dataset_name,
            "task_name": task_name,
            "duckdb_path": str(Path(database_path).resolve()),
            "predictions_path": str(predictions_path),
            "rows_evaluated": len(predictions),
            "batch_size": batch_size,
            "task_type": spec.task_type,
            "target_col": spec.target_col,
            "train_mean_baseline": train_mean,
            "mae": mae,
            "llm_calls_enabled": self.settings.enable_llm_calls,
            "leakage_failures": leakage_failures,
        }
        (output_path / f"{dataset_name}__{task_name}__summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return summary

    def _build_task_spec(self, dataset_name: str, task_name: str, dataset: Any, task: Any) -> RelBenchTaskSpec:
        db = dataset.get_db(upto_test_timestamp=False)
        entity_table = task.entity_table
        candidate_tables = [entity_table]
        for table_name, table in db.table_dict.items():
            if table_name == entity_table:
                continue
            if task.entity_col in table.df.columns:
                candidate_tables.append(table_name)
                continue
            if task.entity_table in table.fkey_col_to_pkey_table.values():
                candidate_tables.append(table_name)
                continue
            if table_name == task.entity_table:
                candidate_tables.append(table_name)
        deduped_tables = list(dict.fromkeys(candidate_tables))
        expected_evidence = list(dict.fromkeys(task_name.replace("-", "_").split("_") + [entity_table, task.entity_col]))
        return RelBenchTaskSpec(
            dataset_name=dataset_name,
            task_name=task_name,
            entity_col=task.entity_col,
            entity_table=task.entity_table,
            time_col=task.time_col,
            target_col=task.target_col,
            task_type=task.task_type.value,
            candidate_tables=deduped_tables,
            expected_evidence=expected_evidence,
        )

    def _run_single_prediction(
        self,
        sample: RuntimeSample,
        task_spec: RelBenchTaskSpec,
        context: DatasetRuntimeContext,
        train_mean: float,
    ) -> dict[str, Any]:
        task = self.engine._build_initial_plan(sample, context)
        evidence = self.engine._gather_initial_evidence(sample, context, task)
        packet = evidence.to_dict()
        packet["benchmark_task"] = {
            "dataset_name": task_spec.dataset_name,
            "task_name": task_spec.task_name,
            "target_col": task_spec.target_col,
            "task_type": task_spec.task_type,
            "entity_col": task_spec.entity_col,
            "cutoff_time": sample.cutoff_time,
        }
        packet["benchmark_baseline"] = {"train_target_mean": train_mean}
        leakage_audit = self._audit_packet_for_leakage(packet, sample.cutoff_time)
        predicted_value, predictor_mode, predictor_prompt, predictor_output = self._predict_numeric(
            packet=packet,
            target_col=task_spec.target_col,
            entity_id=sample.entity_id,
            cutoff_time=sample.cutoff_time,
            baseline_value=train_mean,
        )
        ground_truth = float(sample.ground_truth)
        return {
            "dataset_name": task_spec.dataset_name,
            "task_name": task_spec.task_name,
            "entity_id": sample.entity_id,
            "cutoff_time": sample.cutoff_time,
            "question": sample.question,
            "ground_truth": ground_truth,
            "predicted_value": predicted_value,
            "absolute_error": abs(predicted_value - ground_truth),
            "predictor_mode": predictor_mode,
            "predictor_prompt": predictor_prompt,
            "predictor_output": predictor_output,
            "task_plan": task,
            "evidence_packet": packet,
            "leakage_audit": leakage_audit,
        }

    def _predict_numeric(
        self,
        packet: dict[str, Any],
        target_col: str,
        entity_id: str,
        cutoff_time: str,
        baseline_value: float,
    ) -> tuple[float, str, str, dict[str, Any]]:
        if self.settings.enable_llm_calls:
            try:
                prompt, output = self.predictor.predict_regression(
                    evidence_packet=packet,
                    target_column=target_col,
                    entity_id=entity_id,
                    cutoff_time=cutoff_time,
                )
                predicted = self._coerce_float(output.get("predicted_value"))
                if predicted is not None:
                    return predicted, "llm_regression_predictor", prompt, output
            except Exception as exc:
                return baseline_value, "train_mean_fallback", "", {"error": str(exc)}
        return baseline_value, "train_mean_baseline", "", {"predicted_value": baseline_value}

    def _audit_packet_for_leakage(self, packet: dict[str, Any], cutoff_time: str) -> dict[str, Any]:
        max_seen_time = None
        violations: list[str] = []
        for query in packet.get("queries_run", []):
            result_summary = query.get("result_summary", {})
            sample_rows = result_summary.get("sample_rows", [])
            for row in sample_rows:
                for value in row.values():
                    if self._looks_like_time(value):
                        value_text = str(value)
                        if max_seen_time is None or value_text > max_seen_time:
                            max_seen_time = value_text
                        if value_text > cutoff_time:
                            violations.append(f"{query.get('purpose')}: saw value {value_text} past cutoff {cutoff_time}")
            aggregate_max = result_summary.get("sample_rows", [])
            if aggregate_max:
                continue
        return {
            "passed": not violations,
            "cutoff_time": cutoff_time,
            "max_seen_time": max_seen_time,
            "violations": violations,
        }

    def _render_question(
        self,
        template: Optional[str],
        entity_id: str,
        cutoff_time: str,
        target_col: str,
        entity_col: str,
        row_payload: dict[str, Any],
    ) -> str:
        prompt_template = (
            template
            or "Using only information available on or before {cutoff_time}, predict {target_col} "
            "for {entity_col}={entity_id}."
        )
        values = dict(row_payload)
        values.update(
            {
                "entity_id": entity_id,
                "cutoff_time": cutoff_time,
                "target_col": target_col,
                "entity_col": entity_col,
            }
        )
        return prompt_template.format_map(_SafeFormatDict(values))

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except ValueError:
            return None

    def _looks_like_time(self, value: Any) -> bool:
        if value is None:
            return False
        text = str(value)
        return len(text) >= 10 and text[:4].isdigit() and "-" in text


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
