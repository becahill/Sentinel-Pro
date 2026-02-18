# Architecture

Sentinel-Pro is designed as a small, local-first safety auditing stack.

## Components

- **signals.py**: signal detectors (toxicity, PII, refusal, self-harm, jailbreak, bias)
- **auditor.py**: orchestration + persistence (SQLite)
- **dashboard.py**: Streamlit UI
- **api.py**: FastAPI ingestion service (sync + async queue, rate limiting, readiness probes)
- **web/**: React fullstack web app (control panel)
- **db.py**: SQLAlchemy schema + DB helpers
- **migrations/**: Alembic migrations for Postgres/SQLite
- **gunicorn_conf.py**: production worker, timeout, and structured logging config

## Data flow

1) Input arrives via CLI (CSV/JSONL) or API.
2) `AuditEngine` runs signal detection against the output text.
3) PII is optionally redacted before persistence.
4) Results are written to `audit_logs` in SQLite, including per-signal explanations.
5) `POST /api/audits/async` can enqueue jobs for background workers.
6) The web app and dashboard read from SQLite via the API for metrics + review.
7) Runtime telemetry (latency/error/queue stats) is surfaced via `GET /api/metrics`.

## SQLite schema (summary)

Table: `audit_logs`
- `timestamp`, `input_text`, `output_text`
- `toxicity_score`, `has_pii`, `is_refusal`, `self_harm`, `jailbreak`, `bias`
- `sentiment_score`, `risk_labels`, `risk_explanations`, `pii_types`, `flagged`
- `redaction_applied`, `redaction_count`
- `project_name`, `model_name`, `user_id`, `request_id`, `tags`

## Design intent

- Support local SQLite and production Postgres deployments.
- Keep data inspectable, and easy to extend with migrations.
- Make signals composable while avoiding heavyweight infrastructure.
- Keep operational controls explicit (healthz/readyz, rate limits, queue depth).
