from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from sentinel_pro.core.detectors.base import (
    DetectionResult,
    Detector,
    severity_from_score,
)

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

DEFAULT_TOXICITY_MODEL = "unitary/unbiased-toxic-roberta"
DEFAULT_TOXICITY_THRESHOLD = 0.7


class ToxicityDetector(Detector):
    label = "toxicity"

    def __init__(
        self,
        toxicity_model: Optional[str] = None,
        enable_toxicity: bool = True,
        threshold: float = DEFAULT_TOXICITY_THRESHOLD,
    ) -> None:
        self.toxicity_model = toxicity_model or os.getenv(
            "SENTINEL_TOXICITY_MODEL", DEFAULT_TOXICITY_MODEL
        )
        self.enable_toxicity = (
            enable_toxicity and os.getenv("SENTINEL_DISABLE_TOXICITY", "0") != "1"
        )
        self.threshold = threshold
        self._toxicity_pipeline = None
        self._toxicity_load_error = None
        self._logger = logging.getLogger(__name__)
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def _load_toxicity_pipeline(self) -> None:
        if not self.enable_toxicity:
            return
        if self._toxicity_pipeline is not None or self._toxicity_load_error is not None:
            return
        if not TRANSFORMERS_AVAILABLE:
            self._toxicity_load_error = RuntimeError("transformers is not installed")
            self._logger.warning("transformers is not installed; toxicity disabled.")
            return
        try:
            self._logger.info("Loading toxicity model: %s", self.toxicity_model)
            self._toxicity_pipeline = pipeline(
                "text-classification", model=self.toxicity_model, top_k=None
            )
        except Exception as exc:
            self._toxicity_load_error = exc
            self._logger.warning("Toxicity model unavailable: %s", exc)

    def _iter_results(self, results: Any) -> Iterable[dict]:
        if isinstance(results, list) and results and isinstance(results[0], list):
            return results[0]
        if isinstance(results, list):
            return results
        return []

    def detect_toxicity(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        if not self.enable_toxicity:
            return 0.0
        self._load_toxicity_pipeline()
        if self._toxicity_pipeline is None:
            return 0.0
        try:
            results = self._toxicity_pipeline(text)
            for result in self._iter_results(results):
                label = str(result.get("label", "")).lower()
                if label in {"toxicity", "toxic"}:
                    return float(result.get("score", 0.0))
        except Exception as exc:
            self._logger.warning("Toxicity scoring failed: %s", exc)
        return 0.0

    def detect(self, text: str) -> DetectionResult:
        score = self.detect_toxicity(text)
        detected = score >= self.threshold
        risk_score = score if detected else 0.0
        severity = severity_from_score(risk_score)
        explanation = ""
        if detected:
            explanation = f"toxicity_score {score:.2f} >= {self.threshold:.2f}"
        return DetectionResult(
            label=self.label,
            detected=detected,
            risk_score=risk_score,
            severity=severity,
            explanation=explanation,
            metadata={"toxicity_score": score, "threshold": self.threshold},
        )
