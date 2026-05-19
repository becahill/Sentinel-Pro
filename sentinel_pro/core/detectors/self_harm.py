from __future__ import annotations

import re
from typing import Optional

from sentinel_pro.core.detectors.base import DetectionResult, Detector

SELF_HARM_PATTERN = re.compile(
    r"\b(suicide|kill myself|end my life|self[- ]harm|harm myself|cut myself|overdose|dying|die)\b"
)


class SelfHarmDetector(Detector):
    label = "self_harm"

    def detect_self_harm(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        return bool(SELF_HARM_PATTERN.search(text.lower()))

    def find_self_harm_match(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        match = SELF_HARM_PATTERN.search(text.lower())
        return match.group(0) if match else None

    def detect(self, text: str) -> DetectionResult:
        match = self.find_self_harm_match(text)
        if not match:
            return DetectionResult(label=self.label, detected=False)
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=0.95,
            severity="critical",
            explanation=f"self-harm keyword matched: '{match}'",
            metadata={"matched_phrase": match},
        )
