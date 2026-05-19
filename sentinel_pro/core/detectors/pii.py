from __future__ import annotations

import re
from typing import Dict

from sentinel_pro.core.detectors.base import DetectionResult, Detector

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"


class PiiDetector(Detector):
    label = "pii"

    def detect_pii(self, text: str) -> Dict[str, object]:
        emails = EMAIL_PATTERN.findall(text or "")
        phones = PHONE_PATTERN.findall(text or "")
        pii_data = emails + phones
        pii_types = []
        if emails:
            pii_types.append("email")
        if phones:
            pii_types.append("phone")
        return {
            "has_pii": len(pii_data) > 0,
            "pii_data": pii_data,
            "pii_types": pii_types,
        }

    def redact_pii(self, text: str) -> Dict[str, object]:
        if not isinstance(text, str) or not text:
            return {"redacted_text": text, "redaction_count": 0}
        redacted, email_count = EMAIL_PATTERN.subn(REDACTED_EMAIL, text)
        redacted, phone_count = PHONE_PATTERN.subn(REDACTED_PHONE, redacted)
        return {
            "redacted_text": redacted,
            "redaction_count": email_count + phone_count,
        }

    def detect(self, text: str) -> DetectionResult:
        result = self.detect_pii(text)
        pii_types = result.get("pii_types", [])
        pii_data = result.get("pii_data", [])
        match_count = len(pii_data) if isinstance(pii_data, list) else 0
        detected = bool(result.get("has_pii"))
        if not detected:
            return DetectionResult(label=self.label, detected=False)

        risk_score = 0.85 if match_count > 1 or len(pii_types) > 1 else 0.75
        pii_label = ", ".join(str(item) for item in pii_types) or "unknown"
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=risk_score,
            severity="high",
            explanation=f"PII detected ({pii_label})",
            metadata={"pii_types": pii_types, "match_count": match_count},
        )
