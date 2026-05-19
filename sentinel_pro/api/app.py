from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from celery.exceptions import CeleryError
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from kombu.exceptions import OperationalError as KombuOperationalError
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from sentinel_pro.auth import (
    OAuthTokenResponse,
    extract_key_from_headers,
    issue_oauth_token,
    require_roles,
)
from sentinel_pro.core.detectors import max_severity
from sentinel_pro.jobs.audits import (
    enqueue_audit_task,
    process_audit_payload,
    process_audit_payloads,
    process_audit_task,
)
from sentinel_pro.jobs.audits import (
    payload_to_record as payload_data_to_record,
)
from sentinel_pro.jobs.celery_app import (
    check_celery_broker_ready,
    get_celery_broker_url,
    get_celery_queue_depth,
    get_celery_queue_name,
    is_celery_eager,
)
from sentinel_pro.storage.cache import (
    RedisError,
    cache_get_json,
    cache_set_json,
    get_metrics_cache_ttl_seconds,
    get_redis_client,
    metrics_cache_key,
)
from sentinel_pro.storage.db import get_engine, init_db, resolve_db_url

DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000"
RATE_LIMIT_EXEMPT_PATHS = {"/health", "/healthz", "/readyz"}

LOGGER = logging.getLogger("sentinel.api")


def get_db_url() -> str:
    return resolve_db_url()


def get_webhook_token() -> Optional[str]:
    return os.getenv("SENTINEL_WEBHOOK_TOKEN")


def get_allowed_origins() -> List[str]:
    raw = os.getenv("SENTINEL_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_rate_limit_settings() -> Tuple[int, int]:
    requests = max(1, int(os.getenv("SENTINEL_RATE_LIMIT_REQUESTS", "120")))
    window_seconds = max(1, int(os.getenv("SENTINEL_RATE_LIMIT_WINDOW_SEC", "60")))
    return requests, window_seconds


def get_queue_result_ttl_seconds() -> int:
    return max(60, int(os.getenv("SENTINEL_QUEUE_RESULT_TTL_SEC", "3600")))


def get_enable_toxicity(disable_toxicity: bool) -> bool:
    return (not disable_toxicity) and os.getenv("SENTINEL_DISABLE_TOXICITY", "0") != "1"


def redact_url(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    username = parsed.username or ""
    netloc = f"{username}:***@{host}" if username else host
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_app_logging() -> None:
    level = os.getenv("SENTINEL_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler()
    if os.getenv("SENTINEL_LOG_JSON", "1") == "1":
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])


def init_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
    except Exception:
        LOGGER.warning("SENTRY_DSN set but sentry-sdk is not installed")
        return

    traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
    environment = os.getenv("SENTRY_ENVIRONMENT", "production")
    sentry_sdk.init(
        dsn=dsn,
        integrations=[FastApiIntegration()],
        traces_sample_rate=traces_sample_rate,
        environment=environment,
    )
    log_event(
        "sentry_initialized",
        environment=environment,
        traces_sample_rate=traces_sample_rate,
    )


def log_event(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(fields)
    LOGGER.info(json.dumps(payload, default=str, separators=(",", ":")))


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._request_count = 0
        self._error_count = 0
        self._status_counts: Dict[str, int] = {}
        self._latencies_ms: Deque[float] = deque(maxlen=2000)

    def record(self, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            if status_code >= 500:
                self._error_count += 1
            key = str(status_code)
            self._status_counts[key] = self._status_counts.get(key, 0) + 1
            self._latencies_ms.append(latency_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies_ms)
            request_count = self._request_count
            error_count = self._error_count
            status_counts = dict(self._status_counts)
            uptime_seconds = max(0.0, time.time() - self._started_at)

        return {
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": (error_count / request_count) if request_count else 0.0,
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p95": percentile(latencies, 0.95),
            "status_codes": status_counts,
            "uptime_seconds": uptime_seconds,
        }


ENGINE = get_engine()
RUNTIME_METRICS = RuntimeMetrics()

RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_STATE: Dict[str, Tuple[float, int]] = {}

JOB_METADATA: Dict[str, Tuple[float, Dict[str, Any]]] = {}
JOB_LOCK = threading.Lock()

READ_ROLES = ("admin", "analyst")
WRITE_ROLES = ("admin", "analyst", "ingest")
ADMIN_ROLES = ("admin",)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_app_logging()
    init_sentry()
    init_db(ENGINE)
    log_event(
        "startup_complete",
        queue_backend="celery",
        queue_name=get_celery_queue_name(),
        db_url=get_db_url(),
    )
    try:
        yield
    finally:
        ENGINE.dispose()
        log_event("shutdown_complete")


app = FastAPI(title="Sentinel-Pro API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuditPayload(BaseModel):
    input_text: str = Field(..., description="User input")
    output_text: str = Field(..., description="Model output")
    project_name: Optional[str] = None
    model_name: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    tags: Optional[List[str]] = None
    timestamp: Optional[str] = None


class BatchPayload(BaseModel):
    records: List[AuditPayload]


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * ratio))
    index = max(0, min(index, len(sorted_values) - 1))
    return round(float(sorted_values[index]), 2)


def parse_json_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (TypeError, ValueError):
        pass
    return [str(value)]


def parse_json_array(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (TypeError, ValueError):
        pass
    return []


def parse_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def format_details(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "record_id": details.get("record_id"),
        "flagged": details["flagged"],
        "risk_labels": details["risk_labels"],
        "risk_explanations": details.get("risk_explanations", []),
        "risk_score": details.get("risk_score", 0.0),
        "severity": details.get("severity", "none"),
        "detector_results": details.get("detector_results", []),
        "toxicity_score": details["toxicity_score"],
        "has_pii": details["pii"].get("has_pii"),
        "pii_types": details["pii"].get("pii_types", []),
        "is_refusal": details["is_refusal"],
        "self_harm": details["self_harm"],
        "jailbreak": details["jailbreak"],
        "bias": details["bias"],
        "sentiment_score": details["sentiment_score"],
        "redaction_applied": details.get("redaction_applied", False),
        "redaction_count": details.get("redaction_count", 0),
    }


def verify_webhook_token(token: Optional[str]) -> None:
    expected = get_webhook_token()
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook token")


def get_connection():
    return ENGINE.connect()


def row_to_record(row: Any) -> Dict[str, Any]:
    record = dict(row._mapping)
    record["risk_labels"] = parse_json_list(record.get("risk_labels"))
    record["risk_explanations"] = parse_json_list(record.get("risk_explanations"))
    record["detector_results"] = parse_json_array(record.get("detector_results"))
    record["tags"] = parse_json_list(record.get("tags"))
    record["pii_types"] = parse_json_list(record.get("pii_types"))
    for column in [
        "has_pii",
        "is_refusal",
        "flagged",
        "self_harm",
        "jailbreak",
        "bias",
        "redaction_applied",
    ]:
        record[column] = parse_bool(record.get(column))
    record["redaction_count"] = int(record.get("redaction_count") or 0)
    record["risk_score"] = float(record.get("risk_score") or 0.0)
    record["severity"] = record.get("severity") or "none"
    return record


def build_filters(
    search: Optional[str],
    flagged: Optional[bool],
    project_name: Optional[str],
    model_name: Optional[str],
    user_id: Optional[str],
    tag: Optional[str],
    risk_label: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    clauses = ["1=1"]
    params: Dict[str, Any] = {}
    if flagged is not None:
        clauses.append("flagged = :flagged")
        params["flagged"] = flagged
    if project_name:
        clauses.append("project_name = :project_name")
        params["project_name"] = project_name
    if model_name:
        clauses.append("model_name = :model_name")
        params["model_name"] = model_name
    if user_id:
        clauses.append("user_id = :user_id")
        params["user_id"] = user_id
    if search:
        clauses.append(
            "(LOWER(input_text) LIKE LOWER(:search) OR LOWER(output_text) LIKE LOWER(:search))"
        )
        params["search"] = f"%{search}%"
    if tag:
        for idx, item in enumerate([v.strip() for v in tag.split(",") if v.strip()]):
            key = f"tag_{idx}"
            clauses.append(f"tags LIKE :{key}")
            params[key] = f'%"{item}"%'
    if risk_label:
        for idx, item in enumerate(
            [v.strip() for v in risk_label.split(",") if v.strip()]
        ):
            key = f"risk_{idx}"
            clauses.append(f"risk_labels LIKE :{key}")
            params[key] = f'%"{item}"%'
    if start_date:
        clauses.append("timestamp >= :start_date")
        params["start_date"] = f"{start_date}T00:00:00"
    if end_date:
        clauses.append("timestamp <= :end_date")
        params["end_date"] = f"{end_date}T23:59:59.999999"
    return " AND ".join(clauses), params


def payload_to_record(payload: AuditPayload):
    return payload_data_to_record(payload.model_dump())


def process_payload(payload: AuditPayload, disable_toxicity: bool) -> Dict[str, Any]:
    return process_audit_payload(
        payload.model_dump(), disable_toxicity=disable_toxicity, engine=ENGINE
    )


def get_client_identity(request: Request) -> str:
    key = extract_key_from_headers(
        request.headers.get("x-api-key"), request.headers.get("authorization")
    )
    if key:
        raw = f"key:{key}"
    else:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        client_ip = (
            forwarded_for.split(",")[0].strip()
            if forwarded_for
            else (request.client.host if request.client else "unknown")
        )
        raw = f"ip:{client_ip}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def enforce_rate_limit(request: Request) -> Tuple[bool, int, int, int]:
    limit, window_seconds = get_rate_limit_settings()
    now = time.time()
    identity = get_client_identity(request)

    with RATE_LIMIT_LOCK:
        start, count = RATE_LIMIT_STATE.get(identity, (now, 0))
        if now - start >= window_seconds:
            start = now
            count = 0

        count += 1
        RATE_LIMIT_STATE[identity] = (start, count)

        if len(RATE_LIMIT_STATE) > 50000:
            stale_before = now - (window_seconds * 3)
            stale_keys = [
                key
                for key, (state_start, _) in RATE_LIMIT_STATE.items()
                if state_start < stale_before
            ]
            for key in stale_keys:
                RATE_LIMIT_STATE.pop(key, None)

    reset_seconds = max(0, int(window_seconds - (now - start)))
    remaining = max(0, limit - count)
    allowed = count <= limit
    return allowed, remaining, reset_seconds, limit


def is_exempt_from_rate_limit(path: str) -> bool:
    if path in RATE_LIMIT_EXEMPT_PATHS:
        return True
    if path.startswith("/docs") or path.startswith("/redoc"):
        return True
    if path.startswith("/openapi"):
        return True
    return False


def prune_job_metadata() -> None:
    now = time.time()
    with JOB_LOCK:
        stale_job_ids = [
            job_id
            for job_id, (expires_at, _) in JOB_METADATA.items()
            if expires_at <= now
        ]
        for job_id in stale_job_ids:
            JOB_METADATA.pop(job_id, None)


def job_metadata_key(job_id: str) -> str:
    return f"sentinel-pro:jobs:v1:{job_id}"


def store_job_metadata(job_id: str, metadata: Dict[str, Any]) -> None:
    ttl_seconds = get_queue_result_ttl_seconds()
    payload = {
        "job_id": job_id,
        **metadata,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    client = None if is_celery_eager() else get_redis_client(get_celery_broker_url())
    if client is not None:
        try:
            client.setex(
                job_metadata_key(job_id),
                ttl_seconds,
                json.dumps(payload, default=str, separators=(",", ":")),
            )
        except RedisError:
            pass

    with JOB_LOCK:
        JOB_METADATA[job_id] = (time.time() + ttl_seconds, payload)


def load_job_metadata(job_id: str) -> Optional[Dict[str, Any]]:
    client = None if is_celery_eager() else get_redis_client(get_celery_broker_url())
    if client is not None:
        try:
            payload = client.get(job_metadata_key(job_id))
            if payload:
                return json.loads(payload)
        except (RedisError, TypeError, ValueError):
            pass

    now = time.time()
    with JOB_LOCK:
        item = JOB_METADATA.get(job_id)
        if not item:
            return None
        expires_at, payload = item
        if expires_at <= now:
            JOB_METADATA.pop(job_id, None)
            return None
        return dict(payload)


def queue_stats_snapshot() -> Dict[str, Any]:
    depth = get_celery_queue_depth()
    visible_depth = int(depth or 0)
    return {
        "backend": "celery",
        "queue": get_celery_queue_name(),
        "depth": visible_depth,
        "queued": visible_depth,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "depth_available": depth is not None,
    }


def runtime_snapshot() -> Dict[str, Any]:
    data = RUNTIME_METRICS.snapshot()
    data["queue"] = queue_stats_snapshot()
    return data


def celery_state_to_api_status(state: str) -> str:
    state = state.upper()
    if state == "SUCCESS":
        return "completed"
    if state == "STARTED":
        return "processing"
    if state == "RETRY":
        return "retrying"
    if state in {"FAILURE", "REVOKED"}:
        return "failed"
    return "queued"


def enqueue_job(payload: AuditPayload, disable_toxicity: bool) -> str:
    prune_job_metadata()

    now = datetime.now(timezone.utc).isoformat()
    try:
        job_id = enqueue_audit_task(payload.model_dump(), disable_toxicity)
    except (CeleryError, KombuOperationalError, OSError) as exc:
        log_event("queue_submit_failed", error=str(exc))
        raise HTTPException(
            status_code=503, detail="Audit queue is unavailable"
        ) from exc

    store_job_metadata(
        job_id,
        {
            "status": "queued",
            "submitted_at": now,
            "disable_toxicity": disable_toxicity,
        },
    )

    return job_id


def check_db_ready() -> Tuple[bool, Optional[str]]:
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        return False, str(exc)
    return True, None


def render_incident_report(
    records: List[Dict[str, Any]], filters: Dict[str, Any]
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    flagged = [record for record in records if record.get("flagged")]
    risk_counts = Counter(
        label for record in flagged for label in record.get("risk_labels", [])
    )
    severity_counts = Counter(record.get("severity") or "none" for record in flagged)
    highest_severity = max_severity(
        [str(record.get("severity") or "none") for record in flagged]
    )
    max_risk_score = max(
        [float(record.get("risk_score") or 0.0) for record in flagged],
        default=0.0,
    )

    lines = [
        "# Sentinel-Pro Incident Report",
        f"_Generated (UTC): {generated_at}_",
        "",
        "## Summary",
        f"- Records reviewed: {len(records)}",
        f"- Flagged incidents: {len(flagged)}",
        f"- Highest severity: {highest_severity}",
        f"- Max risk score: {max_risk_score:.2f}",
        "",
        "## Filters",
    ]

    if not filters:
        lines.append("- None")
    else:
        for key, value in filters.items():
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## Risk Breakdown"])
    if risk_counts:
        for label, count in risk_counts.most_common():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- No flagged records in selected scope")

    lines.extend(["", "## Severity Breakdown"])
    if severity_counts:
        for severity, count in severity_counts.most_common():
            lines.append(f"- {severity}: {count}")
    else:
        lines.append("- No flagged records in selected scope")

    lines.extend(["", "## Incident Timeline"])
    if not flagged:
        lines.append("- No incidents available.")
    else:
        for record in sorted(
            flagged,
            key=lambda item: item.get("timestamp") or "",
            reverse=True,
        ):
            risk = ", ".join(record.get("risk_labels", [])) or "none"
            severity = record.get("severity") or "none"
            risk_score = float(record.get("risk_score") or 0.0)
            explanation = (
                record.get("risk_explanations", [""])[0]
                if record.get("risk_explanations")
                else ""
            )
            lines.append(
                f"- {record.get('timestamp')} | #{record.get('id')} | {severity} ({risk_score:.2f}) | {risk} | {explanation}"
            )

    lines.extend(["", "## Detailed Flagged Cases"])
    if not flagged:
        lines.append("No flagged cases to report.")
    else:
        for record in flagged:
            lines.extend(
                [
                    f"### Incident #{record.get('id')}",
                    f"- Timestamp: {record.get('timestamp')}",
                    f"- Project: {record.get('project_name') or '-'}",
                    f"- Model: {record.get('model_name') or '-'}",
                    f"- User: {record.get('user_id') or '-'}",
                    f"- Severity: {record.get('severity') or 'none'}",
                    f"- Risk score: {float(record.get('risk_score') or 0.0):.2f}",
                    f"- Risk labels: {', '.join(record.get('risk_labels', [])) or '-'}",
                    "- Explanations:",
                ]
            )
            explanations = record.get("risk_explanations") or []
            if explanations:
                for explanation in explanations:
                    lines.append(f"  - {explanation}")
            else:
                lines.append("  - None")
            lines.extend(
                [
                    "- Input:",
                    f"  ```\n{record.get('input_text', '')}\n  ```",
                    "- Output:",
                    f"  ```\n{record.get('output_text', '')}\n  ```",
                    "",
                ]
            )

    return "\n".join(lines)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    rate_headers: Dict[str, str] = {}
    start = time.perf_counter()

    should_rate_limit = not is_exempt_from_rate_limit(request.url.path)
    if should_rate_limit:
        allowed, remaining, reset, configured_limit = enforce_rate_limit(request)
        rate_headers = {
            "X-RateLimit-Limit": str(configured_limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset),
        }
        if not allowed:
            elapsed_ms = (time.perf_counter() - start) * 1000
            RUNTIME_METRICS.record(429, elapsed_ms)
            log_event(
                "rate_limit_exceeded",
                method=request.method,
                path=request.url.path,
                client=get_client_identity(request),
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={**rate_headers, "Retry-After": str(reset)},
            )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        RUNTIME_METRICS.record(500, elapsed_ms)
        log_event(
            "request_failed",
            method=request.method,
            path=request.url.path,
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    RUNTIME_METRICS.record(response.status_code, elapsed_ms)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    for key, value in rate_headers.items():
        response.headers[key] = value

    log_event(
        "request_complete",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=round(elapsed_ms, 2),
    )
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "sentinel-pro-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/oauth/token", response_model=OAuthTokenResponse)
async def oauth_token(request: Request):
    return await issue_oauth_token(request)


@app.post("/api/auth/token", response_model=OAuthTokenResponse)
async def api_auth_token(request: Request):
    return await issue_oauth_token(request)


@app.get("/healthz")
async def healthz():
    return await health()


@app.get("/readyz")
async def readyz():
    db_ready, db_error = check_db_ready()
    queue_ready, queue_error = check_celery_broker_ready()
    payload = {
        "status": "ready" if db_ready and queue_ready else "degraded",
        "checks": {
            "database": {"ok": db_ready, "error": db_error},
            "queue": {
                "ok": queue_ready,
                "backend": "celery",
                "broker": redact_url(get_celery_broker_url()),
                "queue": get_celery_queue_name(),
                "error": queue_error,
            },
            "queue_workers": {"ok": queue_ready, "count": None},
        },
    }
    status_code = 200 if db_ready and queue_ready else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/api/audits")
async def create_audit(
    payload: AuditPayload,
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    return process_payload(payload, disable_toxicity)


@app.post("/api/audits/batch")
async def create_audit_batch(
    payload: BatchPayload,
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    results = process_audit_payloads(
        [item.model_dump() for item in payload.records],
        disable_toxicity=disable_toxicity,
        engine=ENGINE,
    )
    return {"count": len(results), "results": results}


@app.post("/api/audits/async")
async def create_audit_async(
    payload: AuditPayload,
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    job_id = enqueue_job(payload, disable_toxicity)
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_depth": get_celery_queue_depth() or 0,
        "status_url": f"/api/audits/jobs/{job_id}",
    }


@app.get("/api/audits/jobs/{job_id}")
async def get_audit_job(job_id: str, _auth=Depends(require_roles(READ_ROLES))):
    metadata = load_job_metadata(job_id)
    result = process_audit_task.AsyncResult(job_id)
    try:
        state = result.state
    except Exception as exc:
        log_event("queue_result_lookup_failed", job_id=job_id, error=str(exc))
        raise HTTPException(
            status_code=503, detail="Audit result backend is unavailable"
        ) from exc

    if state == "PENDING" and not metadata:
        raise HTTPException(status_code=404, detail="Audit job not found")

    payload = {
        "job_id": job_id,
        "status": celery_state_to_api_status(state),
        **(metadata or {}),
    }
    payload["status"] = celery_state_to_api_status(state)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    if state == "SUCCESS":
        payload["result"] = result.result
        if result.date_done:
            payload["finished_at"] = result.date_done.isoformat()
    elif state in {"FAILURE", "REVOKED"}:
        payload["error"] = str(result.result)
        if result.date_done:
            payload["finished_at"] = result.date_done.isoformat()
    elif state == "RETRY" and result.info:
        payload["error"] = str(result.info)
    return payload


@app.get("/api/audits")
async def list_audits(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: Optional[str] = None,
    flagged: Optional[bool] = None,
    project_name: Optional[str] = None,
    model_name: Optional[str] = None,
    user_id: Optional[str] = None,
    tag: Optional[str] = None,
    risk_label: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _auth=Depends(require_roles(READ_ROLES)),
):
    where_clause, params = build_filters(
        search,
        flagged,
        project_name,
        model_name,
        user_id,
        tag,
        risk_label,
        start_date,
        end_date,
    )
    conn = get_connection()
    try:
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM audit_logs WHERE {where_clause}"),
            params,
        ).fetchone()[0]
        rows = conn.execute(
            text(
                f"SELECT * FROM audit_logs WHERE {where_clause} "
                "ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
            ),
            {**params, "limit": page_size, "offset": (page - 1) * page_size},
        ).fetchall()
    finally:
        conn.close()

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "results": [row_to_record(row) for row in rows],
    }


@app.get("/api/audits/{audit_id}")
async def get_audit(audit_id: int, _auth=Depends(require_roles(READ_ROLES))):
    conn = get_connection()
    try:
        row = conn.execute(
            text("SELECT * FROM audit_logs WHERE id = :audit_id"),
            {"audit_id": audit_id},
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return row_to_record(row)


def fetch_metrics_from_db() -> Dict[str, Any]:
    conn = get_connection()
    try:
        metrics_row = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN flagged THEN 1 ELSE 0 END) as flagged,
                    AVG(toxicity_score) as avg_toxicity,
                    AVG(CASE WHEN is_refusal THEN 1.0 ELSE 0.0 END) as refusal_rate,
                    AVG(CASE WHEN self_harm THEN 1.0 ELSE 0.0 END) as self_harm_rate,
                    AVG(CASE WHEN jailbreak THEN 1.0 ELSE 0.0 END) as jailbreak_rate,
                    AVG(CASE WHEN bias THEN 1.0 ELSE 0.0 END) as bias_rate,
                    AVG(CASE WHEN has_pii THEN 1.0 ELSE 0.0 END) as pii_rate,
                    AVG(risk_score) as avg_risk_score,
                    MAX(risk_score) as max_risk_score,
                    MAX(timestamp) as latest_timestamp
                FROM audit_logs
                """)).fetchone()

        risk_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        for row in conn.execute(text("SELECT risk_labels FROM audit_logs")):
            for label in parse_json_list(row[0]):
                risk_counts[label] = risk_counts.get(label, 0) + 1

        for row in conn.execute(
            text(
                "SELECT COALESCE(severity, 'none') as severity, COUNT(*) "
                "FROM audit_logs GROUP BY COALESCE(severity, 'none')"
            )
        ):
            severity_counts[row[0] or "none"] = int(row[1] or 0)

        recent_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        recent_count = conn.execute(
            text("SELECT COUNT(*) FROM audit_logs WHERE timestamp >= :recent_cutoff"),
            {"recent_cutoff": recent_cutoff},
        ).fetchone()[0]
    finally:
        conn.close()

    total = metrics_row[0] or 0
    flagged = metrics_row[1] or 0
    return {
        "total": total,
        "flagged": flagged,
        "flagged_rate": (flagged / total) if total else 0.0,
        "avg_toxicity": metrics_row[2] or 0.0,
        "refusal_rate": metrics_row[3] or 0.0,
        "self_harm_rate": metrics_row[4] or 0.0,
        "jailbreak_rate": metrics_row[5] or 0.0,
        "bias_rate": metrics_row[6] or 0.0,
        "pii_rate": metrics_row[7] or 0.0,
        "avg_risk_score": metrics_row[8] or 0.0,
        "max_risk_score": metrics_row[9] or 0.0,
        "latest_timestamp": metrics_row[10],
        "recent_24h": recent_count,
        "risk_counts": risk_counts,
        "severity_counts": severity_counts,
    }


def get_cached_metrics_from_db() -> Dict[str, Any]:
    ttl_seconds = get_metrics_cache_ttl_seconds()
    key = metrics_cache_key(get_db_url())
    cached = cache_get_json(key) if ttl_seconds > 0 else None
    if cached is not None:
        return cached

    metrics = fetch_metrics_from_db()
    cache_set_json(key, metrics, ttl_seconds)
    return metrics


@app.get("/api/metrics")
async def get_metrics(_auth=Depends(require_roles(READ_ROLES))):
    metrics = get_cached_metrics_from_db()
    return {
        **metrics,
        "runtime": runtime_snapshot(),
    }


@app.get("/api/meta")
async def get_meta(_auth=Depends(require_roles(READ_ROLES))):
    conn = get_connection()
    try:
        projects = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT DISTINCT project_name FROM audit_logs "
                    "WHERE project_name != '' AND project_name IS NOT NULL"
                )
            )
        ]
        models = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT DISTINCT model_name FROM audit_logs "
                    "WHERE model_name != '' AND model_name IS NOT NULL"
                )
            )
        ]
        users = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT DISTINCT user_id FROM audit_logs "
                    "WHERE user_id != '' AND user_id IS NOT NULL"
                )
            )
        ]
        tags: set[str] = set()
        risk_labels: set[str] = set()
        for row in conn.execute(text("SELECT tags, risk_labels FROM audit_logs")):
            tags.update(parse_json_list(row[0]))
            risk_labels.update(parse_json_list(row[1]))
    finally:
        conn.close()

    return {
        "projects": sorted(projects),
        "models": sorted(models),
        "users": sorted(users),
        "tags": sorted(tags),
        "risk_labels": sorted(risk_labels),
    }


@app.get("/api/reports/incidents")
async def export_incident_report(
    limit: int = Query(200, ge=1, le=2000),
    output_format: str = Query("markdown", pattern="^(markdown|json)$"),
    flagged_only: bool = Query(True),
    project_name: Optional[str] = None,
    model_name: Optional[str] = None,
    user_id: Optional[str] = None,
    risk_label: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _auth=Depends(require_roles(READ_ROLES)),
):
    flagged = True if flagged_only else None
    where_clause, params = build_filters(
        search=None,
        flagged=flagged,
        project_name=project_name,
        model_name=model_name,
        user_id=user_id,
        tag=None,
        risk_label=risk_label,
        start_date=start_date,
        end_date=end_date,
    )

    conn = get_connection()
    try:
        rows = conn.execute(
            text(
                f"SELECT * FROM audit_logs WHERE {where_clause} "
                "ORDER BY timestamp DESC LIMIT :limit"
            ),
            {**params, "limit": limit},
        ).fetchall()
    finally:
        conn.close()

    records = [row_to_record(row) for row in rows]
    filters = {
        key: value
        for key, value in {
            "flagged_only": flagged_only,
            "project_name": project_name,
            "model_name": model_name,
            "user_id": user_id,
            "risk_label": risk_label,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        }.items()
        if value not in (None, "")
    }

    if output_format == "json":
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(records),
            "filters": filters,
            "records": records,
        }

    report = render_incident_report(records, filters)
    filename = (
        f"incident_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    )
    return Response(
        content=report,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/audit")
async def audit(
    payload: AuditPayload,
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    return await create_audit(payload, disable_toxicity=disable_toxicity)


@app.post("/audit/async")
async def audit_async(
    payload: AuditPayload,
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    return await create_audit_async(payload, disable_toxicity=disable_toxicity)


@app.post("/audit/batch")
async def audit_batch(
    payload: BatchPayload,
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    return await create_audit_batch(payload, disable_toxicity=disable_toxicity)


@app.post("/webhook")
async def webhook(
    payload: AuditPayload,
    x_sentinel_token: Optional[str] = Header(default=None),
    disable_toxicity: bool = False,
    _auth=Depends(require_roles(WRITE_ROLES)),
):
    verify_webhook_token(x_sentinel_token)
    return await create_audit(payload, disable_toxicity=disable_toxicity)


@app.get("/logs")
async def get_logs(
    limit: int = 100,
    flagged: Optional[bool] = None,
    project_name: Optional[str] = None,
    model_name: Optional[str] = None,
    user_id: Optional[str] = None,
    _auth=Depends(require_roles(ADMIN_ROLES)),
):
    limit = min(max(limit, 1), 1000)
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params: Dict[str, Any] = {}

    if flagged is not None:
        query += " AND flagged = :flagged"
        params["flagged"] = flagged
    if project_name:
        query += " AND project_name = :project_name"
        params["project_name"] = project_name
    if model_name:
        query += " AND model_name = :model_name"
        params["model_name"] = model_name
    if user_id:
        query += " AND user_id = :user_id"
        params["user_id"] = user_id

    query += " ORDER BY timestamp DESC LIMIT :limit"
    params["limit"] = limit

    df = pd.read_sql(text(query), ENGINE, params=params)
    if df.empty:
        return {"count": 0, "results": []}

    df["risk_labels"] = df["risk_labels"].apply(parse_json_list)
    if "risk_explanations" in df.columns:
        df["risk_explanations"] = df["risk_explanations"].apply(parse_json_list)
    if "detector_results" in df.columns:
        df["detector_results"] = df["detector_results"].apply(parse_json_array)
    if "risk_score" not in df.columns:
        df["risk_score"] = 0.0
    if "severity" not in df.columns:
        df["severity"] = "none"
    df["tags"] = df["tags"].apply(parse_json_list)
    df["pii_types"] = df["pii_types"].apply(parse_json_list)

    return {"count": len(df), "results": df.to_dict(orient="records")}


@app.get("/export")
async def export_logs(_auth=Depends(require_roles(ADMIN_ROLES))):
    df = pd.read_sql("SELECT * FROM audit_logs", ENGINE)
    csv_data = df.to_csv(index=False)
    return Response(content=csv_data, media_type="text/csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
