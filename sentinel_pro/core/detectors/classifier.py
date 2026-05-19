from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from sentinel_pro.core.detectors.base import clamp_risk_score

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    CLASSIFIER_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None
    CLASSIFIER_TRANSFORMERS_AVAILABLE = False


@dataclass(frozen=True)
class ClassificationResult:
    detected: bool
    score: float
    source: str
    label: str = ""
    explanation: str = ""
    metadata: Dict[str, Any] | None = None


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    for char in (" ", "-", "/"):
        normalized = normalized.replace(char, "_")
    return normalized


def _iter_pipeline_results(results: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(results, list) and results and isinstance(results[0], list):
        results = results[0]
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                yield item


class OptionalTransformerClassifier:
    def __init__(
        self,
        *,
        model_name: str,
        positive_labels: Sequence[str],
        threshold: float,
        detector_name: str,
    ) -> None:
        self.model_name = model_name
        self.positive_labels = {_normalize_label(label) for label in positive_labels}
        self.threshold = threshold
        self.detector_name = detector_name
        self.local_files_only = (
            os.getenv("SENTINEL_CLASSIFIER_LOCAL_FILES_ONLY", "1") != "0"
        )
        self.enabled = os.getenv("SENTINEL_DISABLE_LOCAL_CLASSIFIERS", "0") != "1"
        self._pipeline = None
        self._load_error: Optional[Exception] = None
        self._logger = logging.getLogger(__name__)
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def _load_pipeline(self) -> None:
        if not self.enabled:
            return
        if self._pipeline is not None or self._load_error is not None:
            return
        if not CLASSIFIER_TRANSFORMERS_AVAILABLE:
            self._load_error = RuntimeError("transformers is not installed")
            self._logger.info(
                "transformers is not installed; %s model scoring disabled.",
                self.detector_name,
            )
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            self._pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
            )
        except Exception as exc:
            self._load_error = exc
            self._logger.info(
                "%s model unavailable (%s); using fallback classifier.",
                self.detector_name,
                exc,
            )

    def classify(self, text: str) -> Optional[ClassificationResult]:
        if not isinstance(text, str) or not text.strip():
            return None
        self._load_pipeline()
        if self._pipeline is None:
            return None

        try:
            raw_results = self._pipeline(text, truncation=True)
        except Exception as exc:
            self._logger.warning("%s model scoring failed: %s", self.detector_name, exc)
            return None

        best_label = ""
        best_score = 0.0
        all_scores: Dict[str, float] = {}
        for result in _iter_pipeline_results(raw_results):
            label = _normalize_label(str(result.get("label", "")))
            score = clamp_risk_score(float(result.get("score", 0.0)))
            all_scores[label] = score
            if label in self.positive_labels and score > best_score:
                best_label = label
                best_score = score

        if not best_label:
            return None

        return ClassificationResult(
            detected=best_score >= self.threshold,
            score=best_score,
            source="transformer",
            label=best_label,
            explanation=f"{self.detector_name} model label '{best_label}' scored {best_score:.2f}",
            metadata={"model": self.model_name, "scores": all_scores},
        )


class LlmJudgeClient:
    def __init__(self, *, detector_name: str, positive_label: str) -> None:
        self.detector_name = detector_name
        self.positive_label = positive_label
        self.url = os.getenv("SENTINEL_LLM_JUDGE_URL", "").strip()
        self.api_key = os.getenv("SENTINEL_LLM_JUDGE_API_KEY", "").strip()
        self.timeout = float(os.getenv("SENTINEL_LLM_JUDGE_TIMEOUT_SEC", "2.5"))
        self.enabled = (
            bool(self.url) and os.getenv("SENTINEL_DISABLE_LLM_JUDGE", "0") != "1"
        )
        self._logger = logging.getLogger(__name__)

    def classify(self, text: str) -> Optional[ClassificationResult]:
        if not self.enabled or not REQUESTS_AVAILABLE:
            return None
        if not isinstance(text, str) or not text.strip():
            return None

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "task": self.detector_name,
            "text": text,
            "response_schema": {
                "detected": "boolean",
                "score": "number between 0 and 1",
                "label": self.positive_label,
                "explanation": "short string",
            },
        }
        try:
            response = requests.post(
                self.url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            self._logger.warning("%s judge request failed: %s", self.detector_name, exc)
            return None

        detected = bool(body.get("detected"))
        score = clamp_risk_score(float(body.get("score", 1.0 if detected else 0.0)))
        label = str((body.get("label") or self.positive_label) if detected else "")
        explanation = str(body.get("explanation") or "")
        return ClassificationResult(
            detected=detected,
            score=score,
            source="llm_judge",
            label=label,
            explanation=explanation,
            metadata={"judge_url": self.url},
        )
