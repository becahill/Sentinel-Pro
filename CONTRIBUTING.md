# Contributing

Thank you for your interest in Sentinel-Pro.

## Branch policy

- `main` is the single source of truth for code and docs.
- Open PRs against `main`, and update docs in the same PR when behavior changes.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Style

- Format: `black .`
- Lint: `ruff check .`

## Tests

```bash
SENTINEL_DISABLE_TOXICITY=1 pytest -q
python scripts/evaluate.py --dataset eval/labeled.jsonl --output-json eval/current_metrics.json
python scripts/check_eval_regression.py --baseline eval/baseline_metrics.json --current eval/current_metrics.json
```

## PR checklist

- [ ] Tests pass locally
- [ ] Eval regression gate passes locally
- [ ] Lint/format checks pass
- [ ] README/docs updated if behavior changes
- [ ] No new files committed from `venv/` or `__pycache__/`
