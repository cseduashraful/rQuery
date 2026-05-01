from __future__ import annotations

from typing import Any


class GroundTruthEvaluator:
    def evaluate(self, answer: dict[str, Any], ground_truth: Any) -> dict[str, Any]:
        truth_text = str(ground_truth).strip().lower()
        if not truth_text:
            return {"score": 0.0, "matched": False, "reason": "empty_ground_truth"}

        answer_text = " ".join(
            [
                str(answer.get("summary", "")),
                str(answer.get("narrative_answer", "")),
                " ".join(str(item) for item in answer.get("ranked_entities", [])),
            ]
        ).lower()
        matched = truth_text in answer_text
        return {
            "score": 1.0 if matched else 0.0,
            "matched": matched,
            "reason": "substring_match" if matched else "ground_truth_not_found",
        }

