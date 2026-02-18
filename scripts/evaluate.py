#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditor import TOXICITY_THRESHOLD  # noqa: E402
from signals import SignalDetector  # noqa: E402

SIGNALS = ["toxicity", "pii", "refusal", "self_harm", "jailbreak", "bias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate signal precision/recall.")
    parser.add_argument("--dataset", required=True, help="Path to labeled JSONL")
    parser.add_argument(
        "--enable-toxicity",
        action="store_true",
        help="Enable toxicity model scoring (slower)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TOXICITY_THRESHOLD,
        help="Toxicity threshold",
    )
    parser.add_argument(
        "--output-json",
        help="Write machine-readable metrics JSON to this path",
    )
    return parser.parse_args()


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_dataset(
    dataset_path: Path, enable_toxicity: bool, threshold: float
) -> Dict[str, Any]:
    detector = SignalDetector(enable_toxicity=enable_toxicity)
    confusion = {signal: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for signal in SIGNALS}

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            output_text = record.get("output_text", "")
            labels = {
                signal: bool(record.get(f"label_{signal}", False)) for signal in SIGNALS
            }

            pii_result = detector.detect_pii(output_text)
            preds = {
                "toxicity": detector.detect_toxicity(output_text) >= threshold,
                "pii": pii_result.get("has_pii", False),
                "refusal": detector.detect_refusal(output_text),
                "self_harm": detector.detect_self_harm(output_text),
                "jailbreak": detector.detect_jailbreak(output_text),
                "bias": detector.detect_bias(output_text),
            }

            for signal in SIGNALS:
                if signal == "toxicity" and not enable_toxicity:
                    continue
                pred = preds[signal]
                label = labels[signal]
                if pred and label:
                    confusion[signal]["tp"] += 1
                elif pred and not label:
                    confusion[signal]["fp"] += 1
                elif not pred and label:
                    confusion[signal]["fn"] += 1
                else:
                    confusion[signal]["tn"] += 1

    per_signal: Dict[str, Dict[str, Any]] = {}
    total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for signal in SIGNALS:
        if signal == "toxicity" and not enable_toxicity:
            per_signal[signal] = {
                "skipped": True,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            continue

        metrics = compute_metrics(**confusion[signal])
        per_signal[signal] = {"skipped": False, **metrics}
        for key in total:
            total[key] += confusion[signal][key]

    combined = compute_metrics(**total)
    return {
        "dataset": str(dataset_path),
        "enable_toxicity": enable_toxicity,
        "threshold": threshold,
        "signals": per_signal,
        "combined": combined,
    }


def print_human_readable(result: Dict[str, Any]) -> None:
    print("Signal evaluation")
    for signal in SIGNALS:
        metrics = result["signals"][signal]
        if metrics.get("skipped"):
            print(f"- {signal}: skipped (enable with --enable-toxicity)")
            continue

        print(
            f"- {signal}: precision={metrics['precision']:.2f} "
            f"recall={metrics['recall']:.2f} f1={metrics['f1']:.2f} "
            f"(tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']})"
        )

    combined = result["combined"]
    print(
        "\nConfusion summary (all signals combined): "
        f"precision={combined['precision']:.2f} "
        f"recall={combined['recall']:.2f} f1={combined['f1']:.2f} "
        f"(tp={combined['tp']} fp={combined['fp']} tn={combined['tn']} fn={combined['fn']})"
    )


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    result = evaluate_dataset(
        dataset_path=dataset_path,
        enable_toxicity=args.enable_toxicity,
        threshold=args.threshold,
    )
    print_human_readable(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWrote metrics JSON to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
