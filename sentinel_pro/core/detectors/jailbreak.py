from __future__ import annotations

from typing import Optional

from sentinel_pro.core.detectors.base import DetectionResult, Detector

JAILBREAK_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "you are now",
    "do anything now",
    "dan",
    "developer mode",
    "jailbreak",
    "system prompt",
    "bypass safety",
)


class JailbreakDetector(Detector):
    label = "jailbreak"

    def detect_jailbreak(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        return any(phrase in lower_text for phrase in JAILBREAK_PHRASES)

    def find_jailbreak_phrase(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        lower_text = text.lower()
        for phrase in JAILBREAK_PHRASES:
            if phrase in lower_text:
                return phrase
        return None

    def detect(self, text: str) -> DetectionResult:
        phrase = self.find_jailbreak_phrase(text)
        if not phrase:
            return DetectionResult(label=self.label, detected=False)
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=0.85,
            severity="high",
            explanation=f"jailbreak phrase matched: '{phrase}'",
            metadata={"matched_phrase": phrase},
        )
