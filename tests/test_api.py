import importlib
import time
from typing import Dict

from fastapi.testclient import TestClient


def make_api(tmp_path, monkeypatch, extra_env: Dict[str, str] | None = None):
    db_path = tmp_path / "audit_logs.db"
    monkeypatch.setenv("SENTINEL_DB_PATH", str(db_path))
    monkeypatch.setenv("SENTINEL_DISABLE_TOXICITY", "1")
    monkeypatch.setenv(
        "SENTINEL_API_KEYS", "admin:test-admin,analyst:test-key,ingest:test-ingest"
    )
    monkeypatch.setenv("SENTINEL_RATE_LIMIT_REQUESTS", "120")
    monkeypatch.setenv("SENTINEL_RATE_LIMIT_WINDOW_SEC", "60")
    monkeypatch.setenv("SENTINEL_QUEUE_WORKERS", "1")
    monkeypatch.setenv("SENTINEL_QUEUE_MAX_SIZE", "100")
    monkeypatch.setenv("SENTINEL_QUEUE_RESULT_TTL_SEC", "3600")

    if extra_env:
        for key, value in extra_env.items():
            monkeypatch.setenv(key, value)

    import api

    return importlib.reload(api)


def test_api_audit_flow(tmp_path, monkeypatch):
    api = make_api(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer test-key"}

    with TestClient(api.app) as client:
        payload = {
            "input_text": "Hello",
            "output_text": "Contact admin@corp.com",
            "project_name": "demo",
            "model_name": "gpt-4o-mini",
            "user_id": "user-001",
            "tags": ["pii"],
        }

        response = client.post(
            "/api/audits?disable_toxicity=true", json=payload, headers=headers
        )
        assert response.status_code == 200
        details = response.json()
        assert details["flagged"] is True

        metrics = client.get("/api/metrics", headers=headers).json()
        assert metrics["total"] == 1
        assert metrics["flagged"] == 1
        assert "runtime" in metrics
        assert "latency_ms_p95" in metrics["runtime"]

        audits = client.get("/api/audits", headers=headers).json()
        assert audits["total"] == 1
        assert audits["results"][0]["project_name"] == "demo"


def test_api_async_queue_job_lifecycle(tmp_path, monkeypatch):
    api = make_api(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer test-key"}

    with TestClient(api.app) as client:
        queued = client.post(
            "/api/audits/async?disable_toxicity=true",
            json={
                "input_text": "hi",
                "output_text": "email me at analyst@corp.com",
                "project_name": "queue",
            },
            headers=headers,
        )
        assert queued.status_code == 200
        job_id = queued.json()["job_id"]

        status_payload = None
        for _ in range(30):
            status_response = client.get(f"/api/audits/jobs/{job_id}", headers=headers)
            assert status_response.status_code == 200
            status_payload = status_response.json()
            if status_payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["status"] == "completed"
        assert status_payload["result"]["flagged"] is True


def test_api_rate_limit_enforced(tmp_path, monkeypatch):
    api = make_api(
        tmp_path,
        monkeypatch,
        extra_env={
            "SENTINEL_RATE_LIMIT_REQUESTS": "1",
            "SENTINEL_RATE_LIMIT_WINDOW_SEC": "60",
        },
    )
    headers = {"Authorization": "Bearer test-key"}
    payload = {"input_text": "hello", "output_text": "safe output"}

    with TestClient(api.app) as client:
        first = client.post(
            "/api/audits?disable_toxicity=true", json=payload, headers=headers
        )
        assert first.status_code == 200

        second = client.post(
            "/api/audits?disable_toxicity=true", json=payload, headers=headers
        )
        assert second.status_code == 429
        assert second.json()["detail"] == "Rate limit exceeded"


def test_health_and_ready_endpoints(tmp_path, monkeypatch):
    api = make_api(tmp_path, monkeypatch)

    with TestClient(api.app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ready = client.get("/readyz")
        assert ready.status_code == 200
        body = ready.json()
        assert body["checks"]["database"]["ok"] is True
        assert body["checks"]["queue_workers"]["ok"] is True


def test_incident_report_export(tmp_path, monkeypatch):
    api = make_api(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer test-key"}

    with TestClient(api.app) as client:
        create = client.post(
            "/api/audits?disable_toxicity=true",
            json={
                "input_text": "Prompt",
                "output_text": "Reach me at security@corp.com",
                "project_name": "incident-demo",
            },
            headers=headers,
        )
        assert create.status_code == 200

        report = client.get("/api/reports/incidents", headers=headers)
        assert report.status_code == 200
        assert "text/markdown" in report.headers["content-type"]
        assert "Sentinel-Pro Incident Report" in report.text

        report_json = client.get(
            "/api/reports/incidents?output_format=json", headers=headers
        )
        assert report_json.status_code == 200
        payload = report_json.json()
        assert payload["count"] >= 1
