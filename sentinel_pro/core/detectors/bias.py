from __future__ import annotations

from typing import Optional

from sentinel_pro.core.detectors.base import DetectionResult, Detector

BIAS_PROTECTED_GROUPS = (
    "women",
    "men",
    "black",
    "white",
    "asian",
    "hispanic",
    "latino",
    "muslim",
    "jewish",
    "christian",
    "gay",
    "lesbian",
    "trans",
    "immigrants",
    "disabled",
)
BIAS_NEGATIVE_DESCRIPTORS = (
    "inferior",
    "stupid",
    "lazy",
    "criminal",
    "dirty",
    "violent",
    "evil",
    "worthless",
    "should be banned",
    "should be deported",
)


class BiasDetector(Detector):
    label = "bias"

    def detect_bias(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        group_hit = any(group in lower_text for group in BIAS_PROTECTED_GROUPS)
        negative_hit = any(term in lower_text for term in BIAS_NEGATIVE_DESCRIPTORS)
        return group_hit and negative_hit

    def find_bias_match(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        lower_text = text.lower()
        group = next((g for g in BIAS_PROTECTED_GROUPS if g in lower_text), None)
        negative = next((t for t in BIAS_NEGATIVE_DESCRIPTORS if t in lower_text), None)
        if group and negative:
            return f"{group} + {negative}"
        return None

    def detect(self, text: str) -> DetectionResult:
        match = self.find_bias_match(text)
        if not match:
            return DetectionResult(label=self.label, detected=False)
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=0.8,
            severity="high",
            explanation=f"bias match: '{match}'",
            metadata={"matched_phrase": match},
        )
