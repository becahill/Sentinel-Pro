# PII Retention and Redaction Policy

## Scope

This policy covers data handled by Sentinel-Pro audit ingestion and storage paths (`auditor.py`, `api.py`, `db.py`).

## Default behavior

- `SENTINEL_REDACT_PII=1` by default.
- Detected PII in `output_text` is redacted before persistence.
- Stored metadata includes:
  - `has_pii` (boolean)
  - `pii_types` (detected type categories)
  - `redaction_applied` and `redaction_count`

Raw detected PII should not be retained in persisted audit outputs when redaction is enabled.

## Data retained per record

Persisted fields include audit context and derived safety signals:

- Request metadata: `project_name`, `model_name`, `user_id`, `request_id`, `tags`, `timestamp`
- Safety signals: toxicity/refusal/self-harm/jailbreak/bias/PII indicators
- Explanations: `risk_explanations`

## Data minimization requirements

- Send only necessary metadata to Sentinel-Pro.
- Do not include secrets in `input_text` / `output_text` payloads.
- Use narrow retention windows in downstream DB backups and exports.

## Configuration controls

- `SENTINEL_REDACT_PII=1` keep enabled in production.
- `SENTINEL_AUTH_REQUIRED=1` to prevent unauthenticated data access.
- `SENTINEL_API_KEYS` with least-privilege roles.

## Operator responsibilities

- Review exported CSV/report artifacts before sharing externally.
- Apply key rotation and access revocation for departing users/services.
- Validate that logs/monitoring do not include raw payload secrets.

## Exception handling

If redaction must be disabled for a controlled debugging scenario (`--no-redact` or `SENTINEL_REDACT_PII=0`), all of the following are required:

1. Temporary non-production environment.
2. Explicit approval from the data owner.
3. Time-boxed retention and post-run purge.
4. Incident review before re-enabling production ingestion.
