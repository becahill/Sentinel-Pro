from __future__ import annotations

import logging
from typing import Dict, Optional

from sentinel_pro.core.detectors import (
    BIAS_NEGATIVE_DESCRIPTORS,
    BIAS_PROTECTED_GROUPS,
    DEFAULT_TOXICITY_MODEL,
    DEFAULT_TOXICITY_THRESHOLD,
    EMAIL_PATTERN,
    JAILBREAK_PHRASES,
    PHONE_PATTERN,
    REDACTED_EMAIL,
    REDACTED_PHONE,
    REFUSAL_PHRASES,
    SELF_HARM_PATTERN,
    TRANSFORMERS_AVAILABLE,
    BiasDetector,
    DetectionResult,
    Detector,
    JailbreakDetector,
    PiiDetector,
    RefusalDetector,
    SelfHarmDetector,
    ToxicityDetector,
)

try:
    from textblob import TextBlob

    try:
        from textblob.exceptions import MissingCorpusError
    except Exception:  # pragma: no cover
        MissingCorpusError = Exception
    TEXTBLOB_AVAILABLE = True
except Exception:  # pragma: no cover
    TextBlob = None
    MissingCorpusError = Exception
    TEXTBLOB_AVAILABLE = False


class SignalDetector:
    def __init__(
        self,
        toxicity_model: Optional[str] = None,
        enable_toxicity: bool = True,
        toxicity_threshold: float = DEFAULT_TOXICITY_THRESHOLD,
    ):
        self._toxicity_detector = ToxicityDetector(
            toxicity_model=toxicity_model,
            enable_toxicity=enable_toxicity,
            threshold=toxicity_threshold,
        )
        self._pii_detector = PiiDetector()
        self._refusal_detector = RefusalDetector()
        self._self_harm_detector = SelfHarmDetector()
        self._jailbreak_detector = JailbreakDetector()
        self._bias_detector = BiasDetector()
        self.detectors = [
            self._toxicity_detector,
            self._pii_detector,
            self._refusal_detector,
            self._self_harm_detector,
            self._jailbreak_detector,
            self._bias_detector,
        ]
        self.toxicity_model = self._toxicity_detector.toxicity_model
        self.enable_toxicity = self._toxicity_detector.enable_toxicity
        self._sentiment_warned = False
        self._logger = logging.getLogger(__name__)

    def _load_toxicity_pipeline(self) -> None:
        self._toxicity_detector._load_toxicity_pipeline()

    def detect_pii(self, text: str) -> Dict[str, object]:
        """Scans for direct identifiers with Presidio-backed recognizers."""
        return self._pii_detector.detect_pii(text)

    def detect_toxicity(self, text: str) -> float:
        """Returns a float score 0.0 to 1.0. Defaults to 0.0 if disabled."""
        return self._toxicity_detector.detect_toxicity(text)

    def detect_refusal(self, text: str) -> bool:
        """Checks if the model refused to answer (compliance signal)."""
        return self._refusal_detector.detect_refusal(text)

    def find_refusal_phrase(self, text: str) -> Optional[str]:
        return self._refusal_detector.find_refusal_phrase(text)

    def detect_self_harm(self, text: str) -> bool:
        """Heuristic detection of self-harm related content."""
        return self._self_harm_detector.detect_self_harm(text)

    def find_self_harm_match(self, text: str) -> Optional[str]:
        return self._self_harm_detector.find_self_harm_match(text)

    def detect_jailbreak(self, text: str) -> bool:
        """Detect prompt injection / jailbreak attempts."""
        return self._jailbreak_detector.detect_jailbreak(text)

    def find_jailbreak_phrase(self, text: str) -> Optional[str]:
        return self._jailbreak_detector.find_jailbreak_phrase(text)

    def detect_bias(self, text: str) -> bool:
        """Detect biased or hateful language targeting protected classes."""
        return self._bias_detector.detect_bias(text)

    def find_bias_match(self, text: str) -> Optional[str]:
        return self._bias_detector.find_bias_match(text)

    def redact_pii(self, text: str) -> Dict[str, object]:
        """Redact PII from text before persistence."""
        return self._pii_detector.redact_pii(text)

    def detect_sentiment(self, text: str) -> float:
        """Uses TextBlob for simple polarity check (-1 to 1)."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        if not TEXTBLOB_AVAILABLE:
            if not self._sentiment_warned:
                self._logger.warning("textblob is not installed; sentiment disabled.")
                self._sentiment_warned = True
            return 0.0
        try:
            return float(TextBlob(text).sentiment.polarity)
        except MissingCorpusError:
            self._logger.warning("TextBlob corpora missing; sentiment disabled.")
        except Exception as exc:
            self._logger.warning("Sentiment scoring failed: %s", exc)
        return 0.0

    def run_detectors(self, text: str) -> Dict[str, DetectionResult]:
        return {detector.label: detector.detect(text) for detector in self.detectors}

    def analyze_output(self, output_text: str) -> Dict[str, object]:
        """Run all signal detectors on the output text."""
        detection_results = self.run_detectors(output_text)
        toxicity_result = detection_results["toxicity"]
        pii_result = self.detect_pii(output_text)
        return {
            "toxicity_score": float(
                toxicity_result.metadata.get(
                    "toxicity_score", toxicity_result.risk_score
                )
            ),
            "pii": pii_result,
            "is_refusal": detection_results["refusal"].detected,
            "self_harm": detection_results["self_harm"].detected,
            "jailbreak": detection_results["jailbreak"].detected,
            "bias": detection_results["bias"].detected,
            "sentiment_score": self.detect_sentiment(output_text),
            "detector_results": [
                result.to_dict() for result in detection_results.values()
            ],
        }


__all__ = [
    "BIAS_NEGATIVE_DESCRIPTORS",
    "BIAS_PROTECTED_GROUPS",
    "DEFAULT_TOXICITY_MODEL",
    "DEFAULT_TOXICITY_THRESHOLD",
    "BiasDetector",
    "DetectionResult",
    "Detector",
    "EMAIL_PATTERN",
    "JAILBREAK_PHRASES",
    "JailbreakDetector",
    "PHONE_PATTERN",
    "PiiDetector",
    "REDACTED_EMAIL",
    "REDACTED_PHONE",
    "REFUSAL_PHRASES",
    "RefusalDetector",
    "SELF_HARM_PATTERN",
    "SelfHarmDetector",
    "SignalDetector",
    "TEXTBLOB_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "ToxicityDetector",
]
