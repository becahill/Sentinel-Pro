from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from sentinel_pro.core.auditor import (
    TOXICITY_THRESHOLD,
    AuditEngine,
    ConversationRecord,
    normalize_tags,
)
from sentinel_pro.core.signals import SignalDetector
from sentinel_pro.jobs.celery_app import celery_app, get_celery_queue_name
from sentinel_pro.storage.cache import invalidate_metrics_cache
from sentinel_pro.storage.db import get_engine, resolve_db_url

_TASK_ENGINE: Optional[Engine] = None
_TASK_ENGINE_URL: Optional[str] = None


def get_enable_toxicity(disable_toxicity: bool) -> bool:
    return (not disable_toxicity) and os.getenv("SENTINEL_DISABLE_TOXICITY", "0") != "1"


def get_task_engine() -> Engine:
    global _TASK_ENGINE, _TASK_ENGINE_URL
    db_url = resolve_db_url()
    if _TASK_ENGINE is None or _TASK_ENGINE_URL != db_url:
        if _TASK_ENGINE is not None:
            _TASK_ENGINE.dispose()
        _TASK_ENGINE = get_engine()
        _TASK_ENGINE_URL = db_url
    return _TASK_ENGINE


def payload_to_record(payload: Mapping[str, Any]) -> ConversationRecord:
    return ConversationRecord(
        input_text=str(payload.get("input_text") or ""),
        output_text=str(payload.get("output_text") or ""),
        project_name=payload.get("project_name"),
        model_name=payload.get("model_name"),
        user_id=payload.get("user_id"),
        request_id=payload.get("request_id"),
        tags=normalize_tags(payload.get("tags")),
        timestamp=payload.get("timestamp"),
    )


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


def _build_engine(engine: Optional[Engine], disable_toxicity: bool) -> AuditEngine:
    detector = SignalDetector(enable_toxicity=get_enable_toxicity(disable_toxicity))
    return AuditEngine(
        db_url=resolve_db_url(),
        engine=engine or get_task_engine(),
        detector=detector,
        toxicity_threshold=TOXICITY_THRESHOLD,
    )


def process_audit_payload(
    payload_data: Mapping[str, Any],
    disable_toxicity: bool,
    engine: Optional[Engine] = None,
) -> Dict[str, Any]:
    with _build_engine(engine, disable_toxicity) as audit_engine:
        details = audit_engine.process_record_with_details(
            payload_to_record(payload_data)
        )
    invalidate_metrics_cache(resolve_db_url())
    return format_details(details)


def process_audit_payloads(
    payloads: Iterable[Mapping[str, Any]],
    disable_toxicity: bool,
    engine: Optional[Engine] = None,
) -> List[Dict[str, Any]]:
    records = [payload_to_record(payload) for payload in payloads]
    if not records:
        return []

    with _build_engine(engine, disable_toxicity) as audit_engine:
        details_list = audit_engine.process_records_with_details(records)
    invalidate_metrics_cache(resolve_db_url())
    return [format_details(details) for details in details_list]


@celery_app.task(
    bind=True,
    name="sentinel_pro.audit.process",
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": int(os.getenv("SENTINEL_CELERY_MAX_RETRIES", "3"))},
)
def process_audit_task(
    self: Any, payload_data: Dict[str, Any], disable_toxicity: bool = False
) -> Dict[str, Any]:
    return process_audit_payload(payload_data, disable_toxicity)


def enqueue_audit_task(payload_data: Dict[str, Any], disable_toxicity: bool) -> str:
    result = process_audit_task.apply_async(
        args=[payload_data, disable_toxicity],
        queue=get_celery_queue_name(),
    )
    return str(result.id)
