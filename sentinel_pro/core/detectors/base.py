from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Literal, Sequence

Severity = Literal["none", "low", "medium", "high", "critical"]

SEVERITIES: Sequence[Severity] = ("none", "low", "medium", "high", "critical")
SEVERITY_RANK = {severity: index for index, severity in enumerate(SEVERITIES)}


def clamp_risk_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def severity_from_score(score: float) -> Severity:
    score = clamp_risk_score(score)
    if score <= 0.0:
        return "none"
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    if score < 0.9:
        return "high"
    return "critical"


def max_severity(values: Sequence[str]) -> Severity:
    selected: Severity = "none"
    for value in values:
        if value in SEVERITY_RANK and SEVERITY_RANK[value] > SEVERITY_RANK[selected]:
            selected = value  # type: ignore[assignment]
    return selected


@dataclass(frozen=True)
class DetectionResult:
    label: str
    detected: bool
    risk_score: float = 0.0
    severity: Severity = "none"
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_risk_signal: bool = True

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_RANK:
            raise ValueError(f"Unsupported severity: {self.severity}")
        object.__setattr__(self, "risk_score", clamp_risk_score(self.risk_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "detected": self.detected,
            "risk_score": self.risk_score,
            "severity": self.severity,
            "explanation": self.explanation,
            "metadata": self.metadata,
            "is_risk_signal": self.is_risk_signal,
        }

    def model_dump(self) -> Dict[str, Any]:
        return self.to_dict()


class Detector(ABC):
    label: ClassVar[str]
    is_risk_signal: ClassVar[bool] = True

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        raise NotImplementedError
