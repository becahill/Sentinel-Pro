# Sentinel-Pro

![CI](https://github.com/becahill/Sentinel-Pro/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)

Sentinel-Pro is an AI safety observability demo for inspecting LLM outputs before they
become invisible production behavior. It ingests conversations, runs safety signal
detectors, persists explainable audit records, and gives reviewers a CLI, Streamlit
dashboard, FastAPI service, React control panel, incident reports, and a small regression
eval harness.

The project is intentionally compact: it is built to show the shape of a real safety
monitoring stack without pretending to be a full production moderation platform.

## Source of truth

`main` is the canonical branch for code and docs. Onboarding, CI, and release notes should
stay aligned to `main`.

## Screenshots and demo

![Sentinel-Pro demo](assets/demo.gif)

If GIFs are blocked, use the PNG fallback:

![Sentinel-Pro screenshot](assets/demo.png)

Capture notes live in `docs/demo_capture.md`.

Run the golden path locally:

```bash
python3 auditor.py --input-jsonl data/golden_path.jsonl --project golden-path --tags demo,golden
streamlit run dashboard.py
```

## Why this matters

LLM applications need an audit trail for model behavior, not just application logs.
Sentinel-Pro demonstrates the observability primitives that make safety review practical:

- signal-level labels for toxicity, PII, refusal, self-harm, jailbreak, and bias
- explainable risk records with matched phrases, thresholds, and redaction metadata
- role-scoped API access for ingestion, review, and export workflows
- dashboards and reports that help humans triage incidents instead of reading raw logs
- regression checks that catch detector behavior changes before they ship

## Quickstart

### Local CLI and Streamlit

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 auditor.py --demo
streamlit run dashboard.py
```

### Fullstack local app

```bash
make install
export SENTINEL_API_KEYS=admin:local-admin,analyst:local-analyst,ingest:local-ingest
export SENTINEL_DB_URL=postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel
python -m alembic upgrade head
make api
make web-install
make web-dev
```

The web UI defaults to `http://localhost:5173` and talks to the API at
`http://localhost:8000`. Override the API URL with `VITE_API_URL`.

Paste `local-admin` into the web app API key field when using the example keys above.

### Docker quickstart

```bash
cp .env.example .env
docker compose up --build
```

Docker Compose starts Postgres, the FastAPI service, the React web app behind nginx, and
the Streamlit dashboard. By default only `http://localhost` is exposed; the API and
dashboard stay internal to the Compose network.

Useful Docker variants:

```bash
# If port 80 is unavailable, change the web port mapping to 8080:80.

# TLS termination with certs in deploy/certs/fullchain.pem and deploy/certs/privkey.pem
docker compose -f docker-compose.yml -f docker-compose.tls.yml up --build

# Localhost/private-network binding with nginx IP allow-list rules
docker compose -f docker-compose.yml -f docker-compose.internal.yml up --build
```

Set `SENTINEL_API_KEYS` in `.env` before using Docker in anything beyond a throwaway local
demo.

## Architecture

```mermaid
graph TD
  A[CSV, JSONL, API, webhook] --> B[AuditEngine]
  B --> C[SignalDetector]
  C --> D[Toxicity detector]
  C --> E[PII detector and redaction]
  C --> F[Refusal detector]
  C --> G[Self-harm detector]
  C --> H[Jailbreak detector]
  C --> I[Bias detector]
  B --> J[(SQLite or Postgres audit_logs)]
  K[FastAPI service] --> B
  K --> L[Async audit queue]
  L --> B
  K --> J
  M[React web app] --> K
  J --> N[Streamlit dashboard]
  K --> O[Metrics and incident reports]
```

Core package layout:

- `sentinel_pro/core/auditor.py`: orchestration, risk aggregation, persistence writes
- `sentinel_pro/core/signals.py`: detector facade used by CLI, API, and evals
- `sentinel_pro/core/detectors/`: individual detector implementations
- `sentinel_pro/api/app.py`: FastAPI app, auth, rate limits, async queue, reports
- `sentinel_pro/auth/dependencies.py`: role-scoped API key dependencies
- `sentinel_pro/storage/db.py`: SQLAlchemy schema, engine creation, lightweight column sync
- `web/`: React review UI
- `dashboard.py`: Streamlit dashboard
- `scripts/evaluate.py`: regression evaluation harness

More detail: `docs/architecture.md`.

## Risk scoring and severity

Each audit response includes signal booleans, `risk_labels`, `risk_explanations`,
`risk_score`, `severity`, and raw `detector_results`.

- `flagged` becomes true when a risk-triggering label is present: `toxicity`, `pii`,
  `self_harm`, `jailbreak`, or `bias`.
- `refusal` is tracked as a compliance/behavior signal, but it is not treated as a risk
  trigger by itself.
- `risk_score` is the maximum normalized risk score across detected risk-triggering
  signals. It ranges from `0.0` to `1.0`.
- `severity` is the highest severity across detected risk-triggering signals:
  `none`, `low`, `medium`, `high`, or `critical`.
- Toxicity uses the model score when the toxicity model is enabled and the score crosses
  the configured threshold.
- Heuristic detectors use fixed scores today: PII is high, jailbreak is high, bias is
  high, and self-harm is critical.

Current score-to-severity mapping:

| Score range | Severity |
| --- | --- |
| `0.0` | `none` |
| `> 0.0` and `< 0.3` | `low` |
| `>= 0.3` and `< 0.7` | `medium` |
| `>= 0.7` and `< 0.9` | `high` |
| `>= 0.9` | `critical` |

## API examples

Start the API:

```bash
export SENTINEL_API_KEYS=admin:local-admin,analyst:local-analyst,ingest:local-ingest
uvicorn api:app --reload
```

Create an audit:

```bash
curl -X POST "http://localhost:8000/api/audits?disable_toxicity=true" \
  -H "Authorization: Bearer local-admin" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Where should the user send logs?",
    "output_text": "Email them to security@corp.com",
    "project_name": "demo",
    "model_name": "gpt-4o-mini",
    "tags": ["pii", "demo"]
  }'
```

Example response:

```json
{
  "record_id": 1,
  "flagged": true,
  "risk_labels": ["pii"],
  "risk_explanations": ["PII detected (email)"],
  "risk_score": 0.75,
  "severity": "high",
  "toxicity_score": 0.0,
  "has_pii": true,
  "pii_types": ["email"],
  "is_refusal": false,
  "self_harm": false,
  "jailbreak": false,
  "bias": false,
  "redaction_applied": true,
  "redaction_count": 1
}
```

Common endpoints:

- `POST /api/audits`: create one audit record
- `POST /api/audits/batch`: create multiple audit records
- `POST /api/audits/async` and `GET /api/audits/jobs/{job_id}`: queue and inspect async audits
- `GET /api/audits`: list records with filters and pagination
- `GET /api/audits/{id}`: fetch one audit record
- `GET /api/metrics`: aggregate safety and runtime metrics
- `GET /api/meta`: filter values for projects, models, users, tags, and labels
- `GET /api/reports/incidents`: export a Markdown or JSON incident report
- `POST /webhook`: webhook-compatible ingestion with optional `X-Sentinel-Token`
- `GET /logs` and `GET /export`: admin-oriented legacy log views and CSV export

Detailed API docs: `docs/api.md`.

## CLI usage

```bash
# Demo data
python3 auditor.py --demo

# Audit CSV or JSONL
python3 auditor.py --input-csv data/sample_conversations.csv
python3 auditor.py --input-jsonl data/sample_conversations.jsonl

# Target Postgres instead of SQLite
python3 auditor.py --db-url postgresql+psycopg://user:pass@localhost:5432/sentinel --demo

# Add metadata defaults
python3 auditor.py --demo --project demo --model gpt-4o-mini --user-id user-01 --tags demo,pii

# Export persisted audit logs
python3 auditor.py --export-csv exports/audit_logs.csv

# Skip toxicity model download
python3 auditor.py --no-toxicity

# Disable PII redaction before persistence
python3 auditor.py --no-redact
```

## Input formats

Required fields:

- `input_text`
- `output_text`

Optional metadata:

- `project_name`
- `model_name`
- `user_id`
- `request_id`
- `tags`
- `timestamp`

CSV:

```csv
input_text,output_text,project_name,tags
"Generate a fake email.","Try contacting admin@corp.com.","demo","pii,example"
```

JSONL:

```json
{"input_text":"Hello","output_text":"Hi there.","project_name":"demo","tags":["safe"]}
```

## Evaluation harness

Sentinel-Pro includes a lightweight regression harness:

```bash
python scripts/evaluate.py --dataset eval/labeled.jsonl --output-json eval/current_metrics.json
python scripts/check_eval_regression.py --baseline eval/baseline_metrics.json --current eval/current_metrics.json
```

`eval/labeled.jsonl` is a small hand-labeled regression dataset, not a production
benchmark. It exists to catch obvious detector regressions in CI and local development.
Do not use its point metrics as broad claims about real-world safety performance.

Dataset shape:

- 60 records total
- 6 signal-specific positive slices with 8 examples each
- 12 negative controls
- one-vs-rest boolean labels such as `label_pii`, `label_jailbreak`, and `label_bias`
- mostly simple, single-signal examples so precision/recall shifts are easy to inspect

By default, toxicity scoring is skipped because the model download is slow and
environment-dependent. Use `--enable-toxicity` when you explicitly want to include it:

```bash
python scripts/evaluate.py --dataset eval/labeled.jsonl --enable-toxicity
```

## Configuration

Common environment variables:

- `SENTINEL_API_KEYS=admin:local-admin,analyst:local-analyst,ingest:local-ingest`
  configures role-scoped API keys in `role:key` format.
- `SENTINEL_AUTH_REQUIRED=1` requires auth even if no keys are configured.
- `SENTINEL_AUTH_DISABLED=1` disables auth checks for local development only.
- `SENTINEL_DB_URL=postgresql+psycopg://user:pass@host:5432/sentinel` selects Postgres.
- `SENTINEL_DB_PATH=path/to/audit_logs.db` selects SQLite when `SENTINEL_DB_URL` is unset.
- `SENTINEL_REDACT_PII=1` redacts detected PII before persistence.
- `SENTINEL_DISABLE_TOXICITY=1` disables toxicity model scoring.
- `SENTINEL_TOXICITY_MODEL=unitary/unbiased-toxic-roberta` overrides the toxicity model.
- `SENTINEL_WEBHOOK_TOKEN=secret` adds an extra shared secret check to `/webhook`.
- `SENTINEL_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000` configures CORS.
- `VITE_API_URL=http://localhost:8000` points the React app at the API.
- `SENTINEL_RATE_LIMIT_REQUESTS=120` and `SENTINEL_RATE_LIMIT_WINDOW_SEC=60` configure
  API rate limiting.
- `SENTINEL_QUEUE_WORKERS=2`, `SENTINEL_QUEUE_MAX_SIZE=1000`, and
  `SENTINEL_QUEUE_RESULT_TTL_SEC=3600` configure async audit processing.
- `SENTINEL_LOG_JSON=1` and `SENTINEL_LOG_LEVEL=INFO` configure API logs.
- `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, and `SENTRY_TRACES_SAMPLE_RATE` enable optional
  Sentry integration.

## Developer shortcuts

```bash
make install
make lint
make format-check
make test
make eval
make eval-gate
make api
make dashboard
make web-dev
make up
```

## Limitations

- The detectors are intentionally simple. Several signals are keyword, regex, or
  threshold based and will miss nuanced context.
- PII detection currently focuses on email addresses and US-style phone numbers.
- PII redaction is best-effort and should not be treated as a complete privacy control.
- The eval dataset is a regression fixture, not a production benchmark or external
  comparison.
- Toxicity scoring depends on an optional local model download and is disabled in Docker
  by default for speed and reproducibility.
- The React and Streamlit dashboards are review surfaces, not case-management systems.
- This is an observability and auditing layer. It does not guarantee model safety.

## Docs

- `docs/api.md`
- `docs/architecture.md`
- `docs/demo_capture.md`
- `docs/security.md`
- `docs/pii_policy.md`
- `docs/threat_model.md`
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
