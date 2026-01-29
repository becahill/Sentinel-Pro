# Sentinel-Pro: AI Output Auditing Workflow

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/becahill/Sentinel-Pro/actions/workflows/ci.yml/badge.svg)

> Update the CI badge URL with your GitHub username/repo name.

Sentinel-Pro is a lightweight auditing workflow that flags risky LLM outputs using a mix of
ML signals and regex heuristics. It is designed to be easy to run locally while still showing
a realistic "human-in-the-loop" safety layer.

## Features
- **Toxicity Detection:** Uses `unitary/unbiased-toxic-roberta` via Transformers.
- **PII Sweeper:** Regex-based detection for email addresses and phone numbers.
- **Compliance Monitor:** Detects model refusals (e.g., "I cannot assist").
- **Self-Harm + Jailbreak + Bias Heuristics:** Fast keyword detection for high-risk content.
- **Persistence:** Logs all transactions to SQLite with project/model/user metadata.
- **Dashboard:** Streamlit UI with filters, search, drill-down, and CSV upload.
- **API:** FastAPI endpoints for real-time ingestion + webhook support.

## Architecture
- `signals.py`: Core detection logic.
- `auditor.py`: Orchestration engine + database layer.
- `dashboard.py`: Streamlit analytics UI.
- `api.py`: FastAPI ingestion service.

## Quick Start

1. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **(Optional) Enable sentiment**
   ```bash
   python3 -m textblob.download_corpora
   ```

3. **Run a demo audit**
   ```bash
   python3 auditor.py --demo
   ```

4. **Launch the dashboard**
   ```bash
streamlit run dashboard.py
```

## Demo

- `assets/demo.gif` (replace with your real capture)
- Capture guide: `docs/demo_capture.md`

## CLI Usage

```bash
python3 auditor.py --help

# Run demo data (default if no input files are provided)
python3 auditor.py --demo

# Audit a CSV file
python3 auditor.py --input-csv data/sample_conversations.csv

# Audit a JSONL file
python3 auditor.py --input-jsonl data/sample_conversations.jsonl

# Export audit logs to CSV
python3 auditor.py --export-csv exports/audit_logs.csv

# Add metadata defaults
python3 auditor.py --demo --project demo --model gpt-4o-mini --user-id user-01 --tags demo,pii

# Skip toxicity model download (faster)
python3 auditor.py --no-toxicity
```

### Input Formats
- **CSV** must include `input_text` and `output_text` columns.
- **JSONL** must contain `input_text` and `output_text` fields per line.

Optional columns/fields:
- `project_name`, `model_name`, `user_id`, `request_id`, `tags`, `timestamp`

Example CSV:
```csv
input_text,output_text,project_name,tags
"Generate a fake email.","Try contacting admin@corp.com.","demo","pii,example"
```

## Dashboard Highlights
- Search and filter by status, project, model, user, tags, or risk labels
- Drill-down record view
- CSV/JSONL upload to audit new data directly from the UI

## API Server

Start the API:
```bash
uvicorn api:app --reload
```

Example request:
```bash
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"input_text":"Hi","output_text":"Contact me at admin@corp.com"}'
```

Webhook endpoint:
- `POST /webhook` (optional `X-Sentinel-Token` if `SENTINEL_WEBHOOK_TOKEN` is set)

Other endpoints:
- `GET /logs?limit=100&flagged=true`
- `GET /export` (CSV)

## Configuration

Environment variables:
- `SENTINEL_DISABLE_TOXICITY=1` disables the toxicity model.
- `SENTINEL_TOXICITY_MODEL=your-model-name` overrides the default model.
- `SENTINEL_DB_PATH=path/to/audit_logs.db` sets the API DB target.
- `SENTINEL_WEBHOOK_TOKEN=secret` protects the webhook endpoint.

## Testing

```bash
SENTINEL_DISABLE_TOXICITY=1 pytest -q
```

## Notes
- The toxicity model is downloaded on first use and may take a minute on CPU.
- The SQLite DB file defaults to `audit_logs.db` in the repo root.

## Sample Outputs
- `examples/sample_audit_output.csv`
- `examples/sample_audit_output.json`
