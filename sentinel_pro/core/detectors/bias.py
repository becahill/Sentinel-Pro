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

DEFAULT_BIAS_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
DEFAULT_BIAS_THRESHOLD = 0.7

BIAS_PROTECTED_GROUPS = (
    "women",
    "men",
    "black",
    "white",
    "asian",
    "hispanic",
    "latino",
    "latina",
    "muslim",
    "jewish",
    "christian",
    "hindu",
    "sikh",
    "gay",
    "lesbian",
    "bisexual",
    "trans",
    "transgender",
    "nonbinary",
    "immigrants",
    "refugees",
    "disabled",
    "elderly",
    "caste",
)

BIAS_NEGATIVE_DESCRIPTORS = (
    "inferior",
    "stupid",
    "lazy",
    "criminal",
    "criminals",
    "dirty",
    "violent",
    "evil",
    "worthless",
    "dangerous",
    "untrustworthy",
    "should be banned",
    "should be deported",
    "should be excluded",
    "should stay quiet",
)

_GROUP_ALIASES = (
    "black people",
    "blacks?",
    "white people",
    "whites?",
    "asian people",
    "asians?",
    "hispanic people",
    "hispanics?",
    "latino people",
    "latinos?",
    "latina people",
    "latinas?",
    "muslims?",
    "jews?",
    "jewish people",
    "christians?",
    "christian people",
    "hindus?",
    "sikhs?",
    "gay people",
    "lesbians?",
    "bisexual people",
    "trans people",
    "transgender people",
    "nonbinary people",
    "immigrants?",
    "refugees?",
    "disabled people",
    "people with disabilities",
    "elderly people",
    "older people",
    "women",
    "men",
    "caste groups?",
)

_NEGATIVE_TERMS = (
    "inferior",
    "stupid",
    "lazy",
    "criminals?",
    "dirty",
    "violent",
    "evil",
    "worthless",
    "dangerous",
    "untrustworthy",
    "not welcome",
    "should stay quiet",
)

_EXCLUSION_TERMS = (
    "banned",
    "deported",
    "excluded",
    "removed",
    "segregated",
)

_GROUP_PATTERN = r"(?:%s)" % "|".join(sorted(_GROUP_ALIASES, key=len, reverse=True))
_NEGATIVE_PATTERN = r"(?:%s)" % "|".join(_NEGATIVE_TERMS)
_EXCLUSION_PATTERN = r"(?:%s)" % "|".join(_EXCLUSION_TERMS)


@dataclass(frozen=True)
class _BiasRule:
    name: str
    pattern: re.Pattern[str]
    score: float
    explanation: str


BIAS_RULES = (
    _BiasRule(
        "identity_stereotype",
        re.compile(
            rf"\b(?P<group>{_GROUP_PATTERN})\b"
            rf"(?:(?!\bnot\b).){{0,80}}\b(?P<term>{_NEGATIVE_PATTERN})\b",
            re.IGNORECASE,
        ),
        0.78,
        "negative stereotype or insult targeting a protected class",
    ),
    _BiasRule(
        "descriptor_before_identity",
        re.compile(
            rf"\b(?P<term>{_NEGATIVE_PATTERN})\b"
            rf"(?:(?!\bnot\b).){{0,40}}\b(?P<group>{_GROUP_PATTERN})\b",
            re.IGNORECASE,
        ),
        0.74,
        "negative descriptor applied to a protected class",
    ),
    _BiasRule(
        "identity_exclusion",
        re.compile(
            rf"\b(?P<group>{_GROUP_PATTERN})\b"
            rf"(?:(?!\bnot\b).){{0,80}}\b(should|must|need to|ought to)\s+"
            rf"(be\s+)?(?P<term>{_EXCLUSION_PATTERN})\b",
            re.IGNORECASE,
        ),
        0.88,
        "exclusionary or removal language targeting a protected class",
    ),
    _BiasRule(
        "all_identity_are",
        re.compile(
            rf"\b(all\s+)?(?P<group>{_GROUP_PATTERN})\b\s+"
            rf"(are|is|were)\s+(?P<term>{_NEGATIVE_PATTERN})\b",
            re.IGNORECASE,
        ),
        0.82,
        "blanket negative claim about a protected class",
    ),
)


class BiasDetector(Detector):
    label = "bias"

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = (
            threshold
            if threshold is not None
            else float(os.getenv("SENTINEL_BIAS_THRESHOLD", DEFAULT_BIAS_THRESHOLD))
        )
        model_name = os.getenv("SENTINEL_BIAS_MODEL", DEFAULT_BIAS_MODEL)
        self._model_classifier = OptionalTransformerClassifier(
            model_name=model_name,
            positive_labels=(
                "hate",
                "hateful",
                "hate_speech",
                "identity_attack",
                "abusive",
                "offensive",
            ),
            threshold=self.threshold,
            detector_name="bias",
        )
        self._judge = LlmJudgeClient(detector_name="bias", positive_label="bias")

    def _rule_classify(self, text: str) -> ClassificationResult:
        best_rule: Optional[_BiasRule] = None
        best_match: Optional[re.Match[str]] = None
        for rule in BIAS_RULES:
            match = rule.pattern.search(text)
            if not match:
                continue
            prefix = text[max(0, match.start() - 16) : match.start()].lower()
            if re.search(r"\b(not|never|no evidence that|false that)\s+$", prefix):
                continue
            if best_rule is None or rule.score > best_rule.score:
                best_rule = rule
                best_match = match

        if best_rule is None or best_match is None:
            return ClassificationResult(
                detected=False,
                score=0.0,
                source="policy_rules",
                metadata={"threshold": self.threshold},
            )

        group = best_match.groupdict().get("group") or ""
        term = best_match.groupdict().get("term") or ""
        return ClassificationResult(
            detected=best_rule.score >= self.threshold,
            score=best_rule.score,
            source="policy_rules",
            label=best_rule.name,
            explanation=best_rule.explanation,
            metadata={
                "matched_phrase": f"{group} + {term}".strip(" +"),
                "matched_group": group,
                "matched_descriptor": term,
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

    def detect_bias(self, text: str) -> bool:
        return self._classify(text).detected

    def find_bias_match(self, text: str) -> Optional[str]:
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
        explanation = result.explanation or f"bias classifier matched: '{matched}'"
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=max(0.7, result.score),
            severity="high",
            explanation=explanation,
            metadata=metadata,
        )
