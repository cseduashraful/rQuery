from __future__ import annotations

import json


def build_summary_prompt(profile_payload: dict) -> str:
    return (
        "You are a database analyst. Summarize the database from the deterministic profile only.\n"
        "Explain what the tables likely represent, important relationships, and predictive signals.\n"
        "Return a concise plain-text summary.\n\n"
        f"PROFILE:\n{json.dumps(profile_payload, indent=2)}"
    )


def build_task_planning_prompt(question: str, db_summary: str, memories: list[dict]) -> str:
    return (
        "You are a data agent that plans predictive analysis over a DuckDB database.\n"
        "Return valid JSON only.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"DATABASE SUMMARY:\n{db_summary}\n\n"
        f"MEMORIES:\n{json.dumps(memories, indent=2)}"
    )


def build_sql_explorer_prompt(task: dict, schema_memory: list[dict], evidence: dict) -> str:
    return (
        "You are a DuckDB SQL exploration agent.\n"
        "Generate at most 3 safe SELECT or WITH queries and return valid JSON only.\n\n"
        f"TASK:\n{json.dumps(task, indent=2)}\n\n"
        f"SCHEMA MEMORY:\n{json.dumps(schema_memory, indent=2)}\n\n"
        f"EVIDENCE SO FAR:\n{json.dumps(evidence, indent=2)}"
    )


def build_prediction_prompt(packet: dict) -> str:
    return (
        "You are a predictive analyst.\n"
        "Use only the evidence provided. Separate facts from assumptions. Return valid JSON only.\n\n"
        f"EVIDENCE PACKET:\n{json.dumps(packet, indent=2)}"
    )

