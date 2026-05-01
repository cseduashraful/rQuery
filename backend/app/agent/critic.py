from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CriticReport:
    prompt_sufficient: bool
    planner_responsibility_score: float
    predictor_responsibility_score: float
    confidence_in_critique: str
    missing_information: list[str] = field(default_factory=list)
    irrelevant_information: list[str] = field(default_factory=list)
    wrong_assumptions: list[str] = field(default_factory=list)
    additional_evidence_requests: list[dict[str, Any]] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_sufficient": self.prompt_sufficient,
            "planner_responsibility_score": self.planner_responsibility_score,
            "predictor_responsibility_score": self.predictor_responsibility_score,
            "confidence_in_critique": self.confidence_in_critique,
            "missing_information": self.missing_information,
            "irrelevant_information": self.irrelevant_information,
            "wrong_assumptions": self.wrong_assumptions,
            "additional_evidence_requests": self.additional_evidence_requests,
            "comments": self.comments,
        }


class PlannerCritic:
    def judge(
        self,
        final_prompt: str,
        answer: dict[str, Any],
        ground_truth: Any,
        expected_evidence: list[str],
        available_sections: list[str],
    ) -> CriticReport:
        prompt_text = final_prompt.lower()
        answer_text = self._answer_text(answer).lower()
        truth_text = str(ground_truth).lower()

        truth_in_prompt = truth_text and truth_text in prompt_text
        truth_in_answer = truth_text and truth_text in answer_text
        sufficient = bool(truth_in_prompt or truth_in_answer)

        missing_information = [
            item for item in expected_evidence if item not in {section.lower() for section in available_sections}
        ]
        additional_requests = [
            {
                "evidence_type": item,
                "reason": f"The final prompt did not clearly include {item}.",
                "priority": "high" if idx == 0 else "medium",
                "likely_tables": [],
            }
            for idx, item in enumerate(missing_information[:3])
        ]

        if sufficient:
            comments = [
                "The final prompt appears to contain enough information to recover the ground truth.",
            ]
            planner_score = 0.2 if truth_in_answer else 0.35
            predictor_score = 1.0 - planner_score
        else:
            comments = [
                "The final prompt appears insufficient for a reliable recovery of ground truth.",
            ]
            planner_score = 0.8
            predictor_score = 0.2

        return CriticReport(
            prompt_sufficient=sufficient,
            planner_responsibility_score=planner_score,
            predictor_responsibility_score=predictor_score,
            confidence_in_critique="medium",
            missing_information=missing_information,
            wrong_assumptions=[] if sufficient else ["The initial evidence did not fully cover the target signals."],
            additional_evidence_requests=additional_requests,
            comments=comments,
        )

    def _answer_text(self, answer: dict[str, Any]) -> str:
        parts = [str(answer.get("summary", "")), str(answer.get("narrative_answer", ""))]
        ranked_entities = answer.get("ranked_entities", [])
        for item in ranked_entities:
            parts.append(str(item))
        return " ".join(parts)

