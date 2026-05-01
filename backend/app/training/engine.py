from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Optional

from backend.app.agent.planner import TaskPlanner
from backend.app.agent.critic import PlannerCritic
from backend.app.agent.evidence import EvidenceAccumulator
from backend.app.agent.prompt_builder import PromptBuilder
from backend.app.duckdb_tools.profiler import DuckDBProfiler
from backend.app.duckdb_tools.sql_safety import SQLSafetyLayer
from backend.app.training.benchmark_loader import BenchmarkSample
from backend.app.training.evaluator import GroundTruthEvaluator


@dataclass
class DatasetRuntimeContext:
    profile: dict[str, Any]
    summary: str
    schema_tables: list[dict[str, Any]]


class FineTuneExecutionEngine:
    def __init__(self) -> None:
        self.profiler = DuckDBProfiler()
        self.safety = SQLSafetyLayer()
        self.prompt_builder = PromptBuilder()
        self.critic = PlannerCritic()
        self.evaluator = GroundTruthEvaluator()
        self.planner = TaskPlanner()
        self._context_cache: dict[str, DatasetRuntimeContext] = {}

    def ensure_context(self, database_path: str) -> DatasetRuntimeContext:
        if database_path in self._context_cache:
            return self._context_cache[database_path]
        bundle = self.profiler.profile_database(database_path)
        schema_tables = [self._schema_only_table(table) for table in bundle.tables]
        summary_lines = ["Database schema summary:"]
        for table in schema_tables:
            column_names = ", ".join(column["column_name"] for column in table["columns"])
            summary_lines.append(f"- {table['table_name']}: columns [{column_names}]")
        context = DatasetRuntimeContext(
            profile={"tables": bundle.tables, "relationships": bundle.relationships},
            schema_tables=schema_tables,
            summary="\n".join(summary_lines),
        )
        self._context_cache[database_path] = context
        return context

    def run_episode(
        self,
        sample: BenchmarkSample,
        max_refinement_rounds: int,
    ) -> dict[str, Any]:
        context = self.ensure_context(sample.dataset.database_path)
        planner_input = {
            "question": sample.question,
            "database_summary": context.summary,
            "memories": context.schema_tables,
            "cutoff_time": sample.cutoff_time,
            "entity_id": sample.entity_id,
            "task_name": sample.task.name,
        }
        initial_plan = self._build_initial_plan(sample, context)
        evidence = self._gather_initial_evidence(sample, context, initial_plan)
        packet = evidence.to_dict()
        prompt = self.prompt_builder.build_final_prompt(packet)
        answer = self._heuristic_prediction(sample, prompt, packet)
        initial_eval = self.evaluator.evaluate(answer, sample.ground_truth)

        refinement_history = []
        critic_report = self.critic.judge(
            final_prompt=prompt,
            answer=answer,
            ground_truth=sample.ground_truth,
            expected_evidence=sample.task.expected_evidence,
            available_sections=self.prompt_builder.available_sections(packet),
        )
        current_packet = packet
        current_prompt = prompt
        current_answer = answer
        current_eval = initial_eval

        for round_index in range(max_refinement_rounds):
            if critic_report.prompt_sufficient:
                break
            additional = self._fetch_additional_evidence(sample, context, critic_report.additional_evidence_requests)
            if not additional:
                break
            current_packet = self.prompt_builder.build_evidence_packet(current_packet, additional)
            current_prompt = self.prompt_builder.build_final_prompt(current_packet)
            current_answer = self._heuristic_prediction(sample, current_prompt, current_packet)
            current_eval = self.evaluator.evaluate(current_answer, sample.ground_truth)
            refinement_history.append(
                {
                    "round": round_index + 1,
                    "additional_evidence": additional,
                    "evaluation": current_eval,
                }
            )
            critic_report = self.critic.judge(
                final_prompt=current_prompt,
                answer=current_answer,
                ground_truth=sample.ground_truth,
                expected_evidence=sample.task.expected_evidence,
                available_sections=self.prompt_builder.available_sections(current_packet),
            )

        return {
            "dataset_name": sample.dataset.name,
            "task_name": sample.task.name,
            "entity_id": sample.entity_id,
            "cutoff_time": sample.cutoff_time,
            "ground_truth": sample.ground_truth,
            "planner_input": planner_input,
            "initial_plan": initial_plan,
            "initial_evaluation": initial_eval,
            "critic_report": critic_report.to_dict(),
            "final_evaluation": current_eval,
            "initial_prompt": prompt,
            "final_prompt": current_prompt,
            "initial_answer": answer,
            "final_answer": current_answer,
            "evidence_packet": current_packet,
            "refinement_history": refinement_history,
        }

    def _build_initial_plan(self, sample: BenchmarkSample, context: DatasetRuntimeContext) -> dict[str, Any]:
        try:
            plan = self.planner.plan(sample.question, context.summary, context.schema_tables)
        except Exception:
            plan = self._heuristic_task_plan(sample)
        plan.setdefault("cutoff_time", sample.cutoff_time)
        plan.setdefault("candidate_tables", list(sample.task.candidate_tables))
        plan.setdefault("needed_information", list(sample.task.expected_evidence))
        assumptions = plan.setdefault("assumptions", [])
        leakage_note = f"Do not use rows newer than cutoff_time {sample.cutoff_time}."
        if leakage_note not in assumptions:
            assumptions.append(leakage_note)
        return plan

    def _heuristic_task_plan(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {
            "intent": "predictive_query",
            "prediction_type": "benchmark_supervised_prediction",
            "entity": {"name": "entity", "entity_id": sample.entity_id},
            "target": {
                "name": sample.task.ground_truth_column,
                "description": f"Predict {sample.task.ground_truth_column} without using data after cutoff.",
                "time_horizon": "as defined by benchmark task",
            },
            "needed_information": list(sample.task.expected_evidence),
            "candidate_tables": list(sample.task.candidate_tables),
            "assumptions": [
                "Do not use rows newer than the entity cutoff_time.",
                f"Cutoff time for this benchmark entity is {sample.cutoff_time}.",
            ],
            "cutoff_time": sample.cutoff_time,
        }

    def _gather_initial_evidence(
        self,
        sample: BenchmarkSample,
        context: DatasetRuntimeContext,
        task: dict[str, Any],
    ) -> EvidenceAccumulator:
        evidence = EvidenceAccumulator(
            user_question=sample.question,
            database_id=sample.dataset.name,
            database_summary=context.summary,
            task=task,
            relevant_tables=[
                table for table in context.schema_tables if table["table_name"] in sample.task.candidate_tables
            ]
            or context.schema_tables[:3],
            relationships=context.profile["relationships"],
        )
        profile_lookup = {table["table_name"]: table for table in context.profile["tables"]}
        for table in evidence.relevant_tables[:5]:
            profile_table = profile_lookup[table["table_name"]]
            for query in self._build_leakage_safe_queries(profile_table, sample):
                safe_sql = self.safety.apply_row_limit(query["sql"], 50)
                self.safety.validate_readonly_sql(safe_sql)
                result = self.safety.execute_safe_query(sample.dataset.database_path, safe_sql, timeout_seconds=10)
                summary = {"row_count": result.row_count, "sample_rows": result.rows[:5]}
                evidence.add_query_result(query["purpose"], safe_sql, summary)
                evidence.facts[f"{profile_table['table_name']}_{query['fact_key']}"] = summary
        return evidence

    def _build_leakage_safe_queries(self, table: dict[str, Any], sample: BenchmarkSample) -> list[dict[str, str]]:
        table_name = table["table_name"]
        queries = []
        temporal_columns = [
            column["column_name"] for column in table["columns"] if column["stats"]["kind"] == "temporal"
        ]
        entity_columns = [column["column_name"] for column in table["columns"] if column["column_name"].endswith("_id")]
        if temporal_columns:
            col = temporal_columns[0]
            queries.append(
                {
                    "purpose": f"Aggregate {table_name} up to cutoff time",
                    "fact_key": "aggregate_before_cutoff",
                    "sql": (
                        f'SELECT COUNT(*) AS row_count, MIN("{col}") AS min_time, MAX("{col}") AS max_time '
                        f'FROM "{table_name}" WHERE "{col}" <= TIMESTAMP \'{sample.cutoff_time}\''
                    ),
                }
            )
        entity_lookup_column = sample.task.entity_lookup_column or sample.task.entity_id_column
        if entity_lookup_column in entity_columns:
            filters = []
            if temporal_columns:
                filters.append(f'"{temporal_columns[0]}" <= TIMESTAMP \'{sample.cutoff_time}\'')
            filters.append(f'"{entity_lookup_column}" = {self._sql_literal(sample.entity_id)}')
            queries.append(
                {
                    "purpose": f"Fetch entity-specific evidence from {table_name}",
                    "fact_key": "entity_rows",
                    "sql": f'SELECT * FROM "{table_name}" WHERE {" AND ".join(filters)} ORDER BY 1 LIMIT 10',
                }
            )
        elif not queries:
            queries.append(
                {
                    "purpose": f"Inspect stable sample from {table_name}",
                    "fact_key": "sample_rows",
                    "sql": f'SELECT * FROM "{table_name}" LIMIT 10',
                }
            )
        return queries

    def _fetch_additional_evidence(
        self,
        sample: BenchmarkSample,
        context: DatasetRuntimeContext,
        requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        additional = {}
        if not requests:
            return additional
        table_lookup = {table["table_name"]: table for table in context.profile["tables"]}
        for request in requests[:3]:
            evidence_type = request["evidence_type"]
            candidate_table = self._match_table_for_evidence(evidence_type, table_lookup)
            if not candidate_table:
                continue
            queries = self._build_leakage_safe_queries(candidate_table, sample)
            if not queries:
                continue
            safe_sql = self.safety.apply_row_limit(queries[0]["sql"], 20)
            self.safety.validate_readonly_sql(safe_sql)
            result = self.safety.execute_safe_query(sample.dataset.database_path, safe_sql, timeout_seconds=10)
            additional[evidence_type] = {
                "requested_by_critic": True,
                "reason": request.get("reason"),
                "rows": result.rows[:5],
                "row_count": result.row_count,
            }
        return additional

    def _match_table_for_evidence(
        self, evidence_type: str, table_lookup: dict[str, dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        lowered = evidence_type.lower()
        for table_name, table in table_lookup.items():
            if any(token in table_name.lower() for token in lowered.replace("_", " ").split()):
                return table
        return next(iter(table_lookup.values()), None)

    def _heuristic_prediction(self, sample: BenchmarkSample, prompt: str, packet: dict[str, Any]) -> dict[str, Any]:
        inferred_label = self._infer_label_from_packet(packet)
        support = []
        for query in packet.get("queries_run", [])[:3]:
            support.append(query["purpose"])
        summary = f"Prediction for entity {sample.entity_id} at cutoff {sample.cutoff_time}."
        return {
            "answer_type": "heuristic_prediction",
            "summary": summary,
            "ranked_entities": [
                {
                    "entity_id": sample.entity_id,
                    "rank": 1,
                    "prediction": inferred_label or "uncertain",
                    "confidence": "medium" if support else "low",
                    "reasoning": support,
                    "supporting_facts": {"cutoff_time": sample.cutoff_time},
                }
            ],
            "narrative_answer": (
                f"The available evidence up to cutoff time {sample.cutoff_time} was used to form a "
                f"prediction for entity {sample.entity_id}. Predicted outcome: {inferred_label or 'uncertain'}."
            ),
            "limitations": list(packet.get("limitations", [])),
        }

    def _infer_label_from_packet(self, packet: dict[str, Any]) -> str:
        queries = packet.get("queries_run", [])
        prioritized = sorted(
            queries,
            key=lambda item: 0 if "payments" in item.get("purpose", "").lower() else 1,
        )
        for query in prioritized:
            rows = query.get("result_summary", {}).get("sample_rows", [])
            for row in rows:
                if "status" in row and row["status"] is not None:
                    return str(row["status"])
        return ""

    def _sql_literal(self, value: Any) -> str:
        text = str(value)
        if text.isdigit():
            return text
        return "'" + text.replace("'", "''") + "'"

    def _schema_only_table(self, table: dict[str, Any]) -> dict[str, Any]:
        return {
            "table_name": table["table_name"],
            "primary_key_guess": table.get("primary_key_guess"),
            "columns": [
                {
                    "column_name": column["column_name"],
                    "data_type": column["data_type"],
                    "kind": column["stats"]["kind"],
                }
                for column in table["columns"]
            ],
        }
