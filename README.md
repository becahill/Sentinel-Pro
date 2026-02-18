# Sentinel-Pro

![CI](https://github.com/becahill/Sentinel-Pro/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)

Sentinel-Pro is a lightweight, end-to-end safety auditing workflow for LLM outputs. It flags
risk signals (toxicity, PII, refusal, self-harm, jailbreak attempts, bias) and ships with a
professional fullstack web app (React + FastAPI) plus the original Streamlit dashboard.

## Source of truth

`main` is the canonical branch for code and docs. All onboarding, CI, and release instructions in this
README are kept aligned to `main` only.

## Quickstart (local)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 auditor.py --demo
streamlit run dashboard.py
```

## Quickstart (fullstack web app)

```bash
make install
export SENTINEL_API_KEYS=admin:local-admin
export SENTINEL_DB_URL=postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel
python -m alembic upgrade head
make api
make web-install
make web-dev
```

The web UI defaults to `http://localhost:5173` and talks to the API at
`http://localhost:8000` (override with `VITE_API_URL`).

## Demo

![Demo GIF](assets/demo.gif)

If GIFs are blocked, use the PNG fallback:

![Demo PNG](assets/demo.png)

Capture guide: `docs/demo_capture.md`

Golden path demo data (shows per-signal breakdown + explanations):
```bash
python3 auditor.py --input-jsonl data/golden_path.jsonl --project golden-path --tags demo,golden
streamlit run dashboard.py
```

## Architecture

```mermaid
graph TD
  A[Inputs: CSV/JSONL/API] --> B[Audit Engine]
  B --> C[Signals: toxicity / PII / refusal / self-harm / jailbreak / bias]
  C --> D[SQLite: audit_logs.db]
  D --> E[Streamlit Dashboard]
  B --> F[FastAPI Service]
  F --> G[React Web App]
```

More detail: `docs/architecture.md`

## CLI / Dashboard / API / Web

### CLI
```bash
# Demo data (default if no input files are provided)
python3 auditor.py --demo

# Audit a CSV file
python3 auditor.py --input-csv data/sample_conversations.csv

# Audit a JSONL file
python3 auditor.py --input-jsonl data/sample_conversations.jsonl

# Target Postgres instead of SQLite
python3 auditor.py --db-url postgresql+psycopg://user:pass@localhost:5432/sentinel --demo

# Export audit logs to CSV
python3 auditor.py --export-csv exports/audit_logs.csv

# Add metadata defaults
python3 auditor.py --demo --project demo --model gpt-4o-mini --user-id user-01 --tags demo,pii

# Skip toxicity model download (faster)
python3 auditor.py --no-toxicity

# Disable PII redaction (not recommended)
python3 auditor.py --no-redact
```

### Dashboard
```bash
streamlit run dashboard.py
```

### Web App
```bash
make web-dev
```
Paste an API key (for example `local-admin`) into the header field to authenticate requests.

### API
```bash
uvicorn api:app --reload
```

```bash
curl -X POST http://localhost:8000/audit \
  -H "Authorization: Bearer local-admin" \
  -H "Content-Type: application/json" \
  -d '{"input_text":"Hi","output_text":"Contact me at admin@corp.com"}'
```

Other endpoints:
- `POST /webhook` (optional `X-Sentinel-Token`)
- `GET /logs?limit=100&flagged=true`
- `GET /export` (CSV)
- `GET /healthz` and `GET /readyz` (liveness/readiness)
- `POST /api/audits/async` + `GET /api/audits/jobs/{job_id}` (background queue)
- `GET /api/reports/incidents` (exportable incident report)

Web app endpoints:
- `GET /api/audits` (filters + pagination)
- `GET /api/audits/{id}`
- `POST /api/audits`
- `GET /api/metrics`
- `GET /api/meta`

Example scripts: `examples/api_usage.sh`

## Input formats

Required fields:
- `input_text` (string)
- `output_text` (string)

Optional fields:
- `project_name`, `model_name`, `user_id`, `request_id`, `tags`, `timestamp`

CSV example:
```csv
input_text,output_text,project_name,tags
"Generate a fake email.","Try contacting admin@corp.com.","demo","pii,example"
```

JSONL example:
```json
{"input_text":"Hello","output_text":"Hi there.","project_name":"demo","tags":["safe"]}
```

## Configuration

Environment variables:
- `SENTINEL_DISABLE_TOXICITY=1` disables the toxicity model
- `SENTINEL_TOXICITY_MODEL=your-model-name` overrides the default model
- `SENTINEL_DB_URL=postgresql+psycopg://user:pass@host:5432/sentinel` sets the DB target
- `SENTINEL_DB_PATH=path/to/audit_logs.db` sets the SQLite DB target (fallback)
- `SENTINEL_WEBHOOK_TOKEN=secret` protects the webhook endpoint
- `SENTINEL_REDACT_PII=1` enables PII redaction before persistence (default)
- `SENTINEL_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000` sets CORS origins
- `VITE_API_URL=http://localhost:8000` points the web app at the API
- `SENTINEL_API_KEYS=admin:local-admin,analyst:local-analyst,ingest:local-ingest` sets API keys + roles (`role:key` format)
- `SENTINEL_AUTH_REQUIRED=1` forces auth even if no keys are configured
- `SENTINEL_AUTH_DISABLED=1` disables auth checks (local dev only)
- `SENTINEL_AUTO_MIGRATE=1` auto-creates tables if migrations are not applied
- `SENTINEL_RATE_LIMIT_REQUESTS=120` request budget per rate-limit window
- `SENTINEL_RATE_LIMIT_WINDOW_SEC=60` rate-limit window in seconds
- `SENTINEL_QUEUE_WORKERS=2` background audit worker count
- `SENTINEL_QUEUE_MAX_SIZE=1000` max queued async jobs
- `SENTINEL_QUEUE_RESULT_TTL_SEC=3600` in-memory async result retention
- `SENTINEL_LOG_JSON=1` emit structured JSON logs
- `SENTINEL_LOG_LEVEL=INFO` API log level
- `SENTRY_DSN=` optional Sentry DSN for API error reporting
- `SENTRY_ENVIRONMENT=production` Sentry environment tag
- `SENTRY_TRACES_SAMPLE_RATE=0.0` Sentry traces sample rate
- `WEB_CONCURRENCY=2` sets gunicorn worker count

TLS (Docker Compose):
- Place `fullchain.pem` and `privkey.pem` in `deploy/certs` and use `docker-compose.tls.yml`.

## Testing

```bash
SENTINEL_DISABLE_TOXICITY=1 pytest -q
python scripts/evaluate.py --dataset eval/labeled.jsonl --output-json eval/current_metrics.json
python scripts/check_eval_regression.py --baseline eval/baseline_metrics.json --current eval/current_metrics.json
```

## Migrations

```bash
python -m alembic upgrade head
```

## Developer shortcuts (Makefile)

```bash
make install
make lint
make format
make test
make demo
make dashboard
make api
make eval
make eval-gate
make up
```

## Operations and observability

- Structured request logs are emitted in JSON by default.
- API runtime telemetry is available at `GET /api/metrics` under the `runtime` field:
  request count, error count/rate, status-code distribution, latency p50/p95, and queue depth/failures.
- Liveness and readiness are separated (`/healthz`, `/readyz`) for production probes.
- Optional Sentry integration is enabled by setting `SENTRY_DSN`.

## Security posture

- API authentication and RBAC use role-scoped API keys (`admin`, `analyst`, `ingest`).
- PII is redacted before persistence by default (`SENTINEL_REDACT_PII=1`).
- Threat model and security docs:
  - `docs/security.md`
  - `docs/pii_policy.md`
  - `Sentinel-Pro-threat-model.md`

## Evaluation harness

```bash
python3 scripts/evaluate.py --dataset eval/labeled.jsonl
# Optional: enable toxicity model for scoring (slower, requires model download)
python3 scripts/evaluate.py --dataset eval/labeled.jsonl --enable-toxicity
```

Outputs per-signal precision/recall and a confusion summary.

## One command up (Docker)

```bash
docker compose up --build
```

This launches Postgres, the API, the web app on `http://localhost`, and the dashboard. Only port `80` is exposed by default; the API and dashboard stay internal to the Compose network.
If port `80` is unavailable, change the web port mapping to `8080:80`.

Before running, create a `.env` file with required secrets:
```bash
cp .env.example .env
```
Set `SENTINEL_API_KEYS` in `.env` to your production keys.

For TLS termination, provide certs in `deploy/certs` and run:
```bash
docker compose -f docker-compose.yml -f docker-compose.tls.yml up --build
```

To expose the dashboard, add a port mapping for the `dashboard` service (for example `8501:8501`).

For internal-only access (localhost + private networks), use:
```bash
docker compose -f docker-compose.yml -f docker-compose.internal.yml up --build
```
This binds the web app to `127.0.0.1:8080` and applies IP allow-list rules in `deploy/nginx.internal.conf`.
Edit `deploy/nginx.internal.conf` to add your VPN or office CIDR ranges.

## Explainability

Each flagged record stores `risk_explanations` (e.g., threshold or keyword match) and the
dashboard surfaces them in the record details panel.

## Limitations

- Heuristics can miss nuanced harm or produce false positives.
- The toxicity model is downloaded on first use and may be slow on CPU.
- Bias detection is keyword-based and not comprehensive.
- This is an auditing layer, not a safety guarantee.
 - PII is redacted before persistence; raw PII is not retained.

## Design decisions

- **SQLite** for zero-config local storage and easy portability.
- **Streamlit** for fast, inspectable safety dashboards.
- **FastAPI** for simple ingestion and webhook compatibility.

## Project status / roadmap

Status: actively maintained as a portfolio-grade safety tooling demo.

Planned:
- Pluggable detectors (policy-based and classifier-based)
- Structured redaction policies per tenant

## Docs

- `docs/architecture.md`
- `docs/demo_capture.md`
- `docs/threat_model.md`
- `docs/security.md`
- `docs/pii_policy.md`
- `Sentinel-Pro-threat-model.md`

## Sample data

- `data/sample_conversations.csv`
- `data/sample_conversations.jsonl`
- `data/golden_path.jsonl`
- `examples/sample_audit_output.csv`
- `examples/sample_audit_output.json`
- `eval/labeled.jsonl`

## License

MIT License. See `LICENSE`.
