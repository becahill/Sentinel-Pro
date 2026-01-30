# Architecture

Sentinel-Pro is designed as a small, local-first safety auditing stack.

## Components

- **signals.py**: signal detectors (toxicity, PII, refusal, self-harm, jailbreak, bias)
- **auditor.py**: orchestration + persistence (SQLite)
- **dashboard.py**: Streamlit UI
- **api.py**: FastAPI ingestion service

## Data flow

1) Input arrives via CLI (CSV/JSONL) or API.
2) `AuditEngine` runs signal detection against the output text.
3) PII is optionally redacted before persistence.
4) Results are written to `audit_logs` in SQLite, including per-signal explanations.
5) The dashboard reads from SQLite for metrics + review.

## SQLite schema (summary)

Table: `audit_logs`
- `timestamp`, `input_text`, `output_text`
- `toxicity_score`, `has_pii`, `is_refusal`, `self_harm`, `jailbreak`, `bias`
- `sentiment_score`, `risk_labels`, `risk_explanations`, `pii_types`, `flagged`
- `redaction_applied`, `redaction_count`
- `project_name`, `model_name`, `user_id`, `request_id`, `tags`

## Design intent

- Keep data local, inspectable, and easy to extend.
- Make signals composable while avoiding heavyweight infrastructure.
