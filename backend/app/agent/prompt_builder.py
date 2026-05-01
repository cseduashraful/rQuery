from __future__ import annotations

import json
from typing import Any


class PromptBuilder:
    def build_evidence_packet(self, base_packet: dict[str, Any], additional_sections: dict[str, Any]) -> dict[str, Any]:
        packet = dict(base_packet)
        packet.setdefault("additional_evidence", {})
        for key, value in additional_sections.items():
            packet["additional_evidence"][key] = value
        return packet

    def build_final_prompt(self, packet: dict[str, Any]) -> str:
        sections = [
            ("USER QUESTION", packet.get("user_question")),
            ("DATABASE SUMMARY", packet.get("database_summary")),
            ("TASK", packet.get("task")),
            ("RELEVANT TABLES", packet.get("relevant_tables")),
            ("RELATIONSHIPS", packet.get("relationships")),
            ("FACTS", packet.get("facts")),
            ("CANDIDATE ENTITIES", packet.get("candidate_entities")),
            ("CRITIC-REQUESTED EVIDENCE", packet.get("additional_evidence", {})),
            ("LIMITATIONS", packet.get("limitations")),
            ("QUERIES RUN", packet.get("queries_run")),
        ]
        rendered = []
        for label, value in sections:
            if value in (None, [], {}, ""):
                continue
            rendered.append(f"{label}:\n{json.dumps(value, indent=2, default=str)}")
        return (
            "You are a predictive analyst.\n"
            "Use only the evidence below. Separate facts from assumptions.\n\n"
            + "\n\n".join(rendered)
        )

    def available_sections(self, packet: dict[str, Any]) -> list[str]:
        sections = []
        for section in packet.keys():
            if packet.get(section) not in (None, [], {}, ""):
                sections.append(section)
        facts = packet.get("facts", {})
        sections.extend(facts.keys())
        for query in packet.get("queries_run", []):
            purpose = query.get("purpose")
            if purpose:
                sections.append(str(purpose))
        additional = packet.get("additional_evidence", {})
        sections.extend(additional.keys())
        return sections
