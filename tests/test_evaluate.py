from __future__ import annotations

import argparse

import pytest

from scripts.evaluate import metric_gate_failures, metric_threshold


def test_metric_gate_failures_flags_low_non_skipped_metrics() -> None:
    result = {
        "combined": {"precision": 0.93, "recall": 0.91},
        "signals": {
            "toxicity": {"skipped": True, "precision": 0.0, "recall": 0.0},
            "pii": {"skipped": False, "precision": 0.91, "recall": 0.99},
            "refusal": {"skipped": False, "precision": 1.0, "recall": 1.0},
            "self_harm": {"skipped": False, "precision": 1.0, "recall": 1.0},
            "jailbreak": {"skipped": False, "precision": 1.0, "recall": 1.0},
            "bias": {"skipped": False, "precision": 1.0, "recall": 1.0},
        },
    }

    failures = metric_gate_failures(
        result=result,
        min_precision=0.92,
        min_recall=0.92,
    )

    assert failures == [
        "combined.recall=0.9100 below minimum 0.9200",
        "pii.precision=0.9100 below minimum 0.9200",
    ]


def test_metric_threshold_rejects_out_of_range_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        metric_threshold("1.01")
