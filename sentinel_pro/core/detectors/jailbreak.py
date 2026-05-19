from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from sentinel_pro.core.detectors.base import DetectionResult, Detector
from sentinel_pro.core.detectors.classifier import (
    ClassificationResult,
    LlmJudgeClient,
    OptionalTransformerClassifier,
)

DEFAULT_JAILBREAK_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_JAILBREAK_THRESHOLD = 0.72

JAILBREAK_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "developer mode",
    "do anything now",
    "dan mode",
    "jailbreak",
    "system prompt",
    "hidden instructions",
    "bypass safety",
    "disable guardrails",
)


@dataclass(frozen=True)
class _JailbreakRule:
    name: str
    pattern: re.Pattern[str]
    score: float
    explanation: str


JAILBREAK_RULES = (
    _JailbreakRule(
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\s+"
            r"(all\s+)?(previous|prior|above|system|developer)\s+"
            r"(instructions?|rules?|prompts?|messages?)\b",
            re.IGNORECASE,
        ),
        0.9,
        "attempts to override higher-priority instructions",
    ),
    _JailbreakRule(
        "system_prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|dump|leak|share|output)\s+(the\s+)?"
            r"(system|developer|hidden)\s+"
            r"(prompt|message|instructions?|policy|rules?)\b",
            re.IGNORECASE,
        ),
        0.88,
        "attempts to extract hidden system or developer content",
    ),
    _JailbreakRule(
        "system_prompt_leak",
        re.compile(
            r"\b(system|developer|hidden)\s+(prompt|message|instructions?)\s+"
            r"(leak|dump|please|now|verbatim)\b",
            re.IGNORECASE,
        ),
        0.84,
        "asks for hidden prompt leakage",
    ),
    _JailbreakRule(
        "safety_bypass",
        re.compile(
            r"\b(bypass|disable|ignore|turn off|remove)\s+"
            r"(safety|guardrails?|filters?|policy|policies|restrictions?)\b",
            re.IGNORECASE,
        ),
        0.86,
        "attempts to bypass safety controls",
    ),
    _JailbreakRule(
        "unrestricted_roleplay",
        re.compile(
            r"\b(you are now|act as|pretend to be)\s+"
            r"(in\s+)?(developer mode|dan|an unrestricted|a rule[- ]?free|"
            r"an uncensored|a no[- ]?limits)\b",
            re.IGNORECASE,
        ),
        0.8,
        "attempts to move the assistant into an unrestricted role",
    ),
    _JailbreakRule(
        "known_jailbreak_mode",
        re.compile(
            r"\b(jailbreak|do anything now|dan\s+mode|developer\s+mode)\b",
            re.IGNORECASE,
        ),
        0.78,
        "references a known jailbreak mode",
    ),
)


class JailbreakDetector(Detector):
    label = "jailbreak"

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = (
            threshold
            if threshold is not None
            else float(
                os.getenv("SENTINEL_JAILBREAK_THRESHOLD", DEFAULT_JAILBREAK_THRESHOLD)
            )
        )
        model_name = os.getenv("SENTINEL_JAILBREAK_MODEL", DEFAULT_JAILBREAK_MODEL)
        self._model_classifier = OptionalTransformerClassifier(
            model_name=model_name,
            positive_labels=(
                "injection",
                "prompt_injection",
                "jailbreak",
                "malicious",
                "unsafe",
                "attack",
            ),
            threshold=self.threshold,
            detector_name="jailbreak",
        )
        self._judge = LlmJudgeClient(
            detector_name="jailbreak", positive_label="jailbreak"
        )

    def _rule_classify(self, text: str) -> ClassificationResult:
        best_match: Optional[re.Match[str]] = None
        best_rule: Optional[_JailbreakRule] = None
        for rule in JAILBREAK_RULES:
            match = rule.pattern.search(text)
            if match and (best_rule is None or rule.score > best_rule.score):
                best_rule = rule
                best_match = match

        if best_rule is None or best_match is None:
            return ClassificationResult(
                detected=False,
                score=0.0,
                source="policy_rules",
                metadata={"threshold": self.threshold},
            )

        matched_text = best_match.group(0)
        return ClassificationResult(
            detected=best_rule.score >= self.threshold,
            score=best_rule.score,
            source="policy_rules",
            label=best_rule.name,
            explanation=best_rule.explanation,
            metadata={
                "matched_phrase": matched_text,
                "rule": best_rule.name,
                "threshold": self.threshold,
            },
        )

    def _classify(self, text: str) -> ClassificationResult:
        if not isinstance(text, str) or not text.strip():
            return ClassificationResult(False, 0.0, "none")

        candidates = []
        model_result = self._model_classifier.classify(text)
        if model_result is not None:
            candidates.append(model_result)

        judge_result = self._judge.classify(text)
        if judge_result is not None:
            candidates.append(judge_result)

        candidates.append(self._rule_classify(text))
        return max(candidates, key=lambda result: result.score)

    def detect_jailbreak(self, text: str) -> bool:
        return self._classify(text).detected

    def find_jailbreak_phrase(self, text: str) -> Optional[str]:
        result = self._classify(text)
        metadata = result.metadata or {}
        matched = metadata.get("matched_phrase")
        if matched:
            return str(matched)
        if result.detected:
            return result.label or result.source
        return None

    def detect(self, text: str) -> DetectionResult:
        result = self._classify(text)
        if not result.detected:
            return DetectionResult(
                label=self.label,
                detected=False,
                metadata={"source": result.source, "score": result.score},
            )

        metadata = {
            "source": result.source,
            "score": result.score,
            "label": result.label,
            **(result.metadata or {}),
        }
        matched = metadata.get("matched_phrase") or result.label or result.source
        explanation = result.explanation or f"jailbreak classifier matched: '{matched}'"
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=max(0.75, result.score),
            severity="high",
            explanation=explanation,
            metadata=metadata,
        )
