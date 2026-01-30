#!/usr/bin/env bash
set -euo pipefail

# Start the API: uvicorn api:app --reload

curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"input_text":"Hello","output_text":"Contact me at admin@corp.com"}'

python3 - <<'PY'
import requests

payload = {"input_text": "Hello", "output_text": "You are useless."}
resp = requests.post("http://localhost:8000/audit", json=payload, timeout=10)
print(resp.json())
PY
