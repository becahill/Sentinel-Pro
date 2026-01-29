# Contributing

Thanks for your interest in Sentinel-Pro.

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
```

## PR checklist

- [ ] Tests pass locally
- [ ] Lint/format checks pass
- [ ] README/docs updated if behavior changes
- [ ] No new files committed from `venv/` or `__pycache__/`
