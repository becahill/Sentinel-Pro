#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from auditor import TOXICITY_THRESHOLD
from signals import SignalDetector

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
    return parser.parse_args()


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
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


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    detector = SignalDetector(enable_toxicity=args.enable_toxicity)
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
                "toxicity": detector.detect_toxicity(output_text) >= args.threshold,
                "pii": pii_result.get("has_pii", False),
                "refusal": detector.detect_refusal(output_text),
                "self_harm": detector.detect_self_harm(output_text),
                "jailbreak": detector.detect_jailbreak(output_text),
                "bias": detector.detect_bias(output_text),
            }

            for signal in SIGNALS:
                if signal == "toxicity" and not args.enable_toxicity:
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

    print("Signal evaluation")
    for signal in SIGNALS:
        if signal == "toxicity" and not args.enable_toxicity:
            print(f"- {signal}: skipped (enable with --enable-toxicity)")
            continue
        metrics = compute_metrics(**confusion[signal])
        print(
            f"- {signal}: precision={metrics['precision']:.2f} "
            f"recall={metrics['recall']:.2f} f1={metrics['f1']:.2f} "
            f"(tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']})"
        )

    total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for signal in SIGNALS:
        if signal == "toxicity" and not args.enable_toxicity:
            continue
        for key in total:
            total[key] += confusion[signal][key]

    total_metrics = compute_metrics(**total)
    print(
        "\nConfusion summary (all signals combined): "
        f"precision={total_metrics['precision']:.2f} "
        f"recall={total_metrics['recall']:.2f} f1={total_metrics['f1']:.2f} "
        f"(tp={total_metrics['tp']} fp={total_metrics['fp']} tn={total_metrics['tn']} fn={total_metrics['fn']})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
