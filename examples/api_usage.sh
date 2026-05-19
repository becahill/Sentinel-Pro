#!/usr/bin/env bash
set -euo pipefail

# Start the API: uvicorn api:app --reload
export TOKEN="$(
  curl -s -X POST http://localhost:8000/oauth/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials&client_id=admin-cli&client_secret=${SENTINEL_ADMIN_CLIENT_SECRET:-local-admin-secret}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
)"

curl -X POST http://localhost:8000/audit \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"input_text":"Hello","output_text":"Contact me at admin@corp.com"}'

python3 - <<'PY'
import requests
import os

payload = {"input_text": "Hello", "output_text": "You are useless."}
headers = {"Authorization": f"Bearer {os.environ['TOKEN']}"}
resp = requests.post("http://localhost:8000/audit", json=payload, headers=headers, timeout=10)
print(resp.json())
PY
