#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if eval precision/recall regresses versus baseline."
    )
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON path")
    parser.add_argument("--current", required=True, help="Current metrics JSON path")
    parser.add_argument(
        "--allowed-drop",
        type=float,
        default=0.0,
        help="Allowed absolute precision/recall drop before failing",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_metrics(
    baseline: Dict[str, Any], current: Dict[str, Any], allowed_drop: float
) -> List[str]:
    failures: List[str] = []

    baseline_signals = baseline.get("signals", {})
    current_signals = current.get("signals", {})

    for signal, base_metrics in baseline_signals.items():
        curr_metrics = current_signals.get(signal)
        if curr_metrics is None:
            failures.append(f"Signal '{signal}' is missing from current metrics")
            continue

        if base_metrics.get("skipped"):
            continue
        if curr_metrics.get("skipped"):
            failures.append(f"Signal '{signal}' is skipped in current metrics")
            continue

        for metric_name in ("precision", "recall"):
            baseline_value = float(base_metrics.get(metric_name, 0.0))
            current_value = float(curr_metrics.get(metric_name, 0.0))
            drop = baseline_value - current_value
            if drop > allowed_drop:
                failures.append(
                    " ".join(
                        [
                            f"{signal}.{metric_name} regressed:",
                            f"baseline={baseline_value:.4f}",
                            f"current={current_value:.4f}",
                            f"drop={drop:.4f}",
                            f"allowed={allowed_drop:.4f}",
                        ]
                    )
                )

    return failures


def main() -> int:
    args = parse_args()
    baseline = load_json(args.baseline)
    current = load_json(args.current)

    failures = compare_metrics(baseline, current, args.allowed_drop)

    if failures:
        print("Eval regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Eval regression check passed: no precision/recall degradation detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
