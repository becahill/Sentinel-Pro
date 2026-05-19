from __future__ import annotations

from typing import Optional

from sentinel_pro.core.detectors.base import DetectionResult, Detector

REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i am unable to",
    "as an ai",
    "compliance violation",
    "against my programming",
    "i apologize",
    "i will not",
    "i won't",
)


class RefusalDetector(Detector):
    label = "refusal"
    is_risk_signal = False

    def detect_refusal(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        return any(phrase in lower_text for phrase in REFUSAL_PHRASES)

    def find_refusal_phrase(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        lower_text = text.lower()
        for phrase in REFUSAL_PHRASES:
            if phrase in lower_text:
                return phrase
        return None

    def detect(self, text: str) -> DetectionResult:
        phrase = self.find_refusal_phrase(text)
        if not phrase:
            return DetectionResult(
                label=self.label, detected=False, is_risk_signal=self.is_risk_signal
            )
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=0.0,
            severity="none",
            explanation=f"refusal phrase matched: '{phrase}'",
            metadata={"matched_phrase": phrase},
            is_risk_signal=self.is_risk_signal,
        )
