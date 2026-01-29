import json
import os
import sqlite3
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auditor import TOXICITY_THRESHOLD, AuditEngine, ConversationRecord, normalize_tags
from signals import SignalDetector

DB_PATH = os.getenv("SENTINEL_DB_PATH", "audit_logs.db")
WEBHOOK_TOKEN = os.getenv("SENTINEL_WEBHOOK_TOKEN")

app = FastAPI(title="Sentinel-Pro API", version="0.1.0")


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


def parse_json_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [str(value)]


def format_details(details: dict) -> dict:
    return {
        "flagged": details["flagged"],
        "risk_labels": details["risk_labels"],
        "toxicity_score": details["toxicity_score"],
        "has_pii": details["pii"].get("has_pii"),
        "pii_types": details["pii"].get("pii_types", []),
        "is_refusal": details["is_refusal"],
        "self_harm": details["self_harm"],
        "jailbreak": details["jailbreak"],
        "bias": details["bias"],
        "sentiment_score": details["sentiment_score"],
    }


def verify_webhook_token(token: Optional[str]) -> None:
    if WEBHOOK_TOKEN and token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid webhook token")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/audit")
async def audit(payload: AuditPayload, disable_toxicity: bool = False):
    detector = SignalDetector(enable_toxicity=not disable_toxicity)
    record = ConversationRecord(
        input_text=payload.input_text,
        output_text=payload.output_text,
        project_name=payload.project_name,
        model_name=payload.model_name,
        user_id=payload.user_id,
        request_id=payload.request_id,
        tags=normalize_tags(payload.tags),
        timestamp=payload.timestamp,
    )
    with AuditEngine(
        db_path=DB_PATH, detector=detector, toxicity_threshold=TOXICITY_THRESHOLD
    ) as engine:
        details = engine.process_record_with_details(record)
    return format_details(details)


@app.post("/audit/batch")
async def audit_batch(payload: BatchPayload, disable_toxicity: bool = False):
    detector = SignalDetector(enable_toxicity=not disable_toxicity)
    with AuditEngine(
        db_path=DB_PATH, detector=detector, toxicity_threshold=TOXICITY_THRESHOLD
    ) as engine:
        results = []
        for record_payload in payload.records:
            record = ConversationRecord(
                input_text=record_payload.input_text,
                output_text=record_payload.output_text,
                project_name=record_payload.project_name,
                model_name=record_payload.model_name,
                user_id=record_payload.user_id,
                request_id=record_payload.request_id,
                tags=normalize_tags(record_payload.tags),
                timestamp=record_payload.timestamp,
            )
            details = engine.process_record_with_details(record)
            results.append(format_details(details))
    return {"count": len(results), "results": results}


@app.post("/webhook")
async def webhook(
    payload: AuditPayload,
    x_sentinel_token: Optional[str] = Header(default=None),
    disable_toxicity: bool = False,
):
    verify_webhook_token(x_sentinel_token)
    return await audit(payload, disable_toxicity=disable_toxicity)


@app.get("/logs")
async def get_logs(
    limit: int = 100,
    flagged: Optional[bool] = None,
    project_name: Optional[str] = None,
    model_name: Optional[str] = None,
    user_id: Optional[str] = None,
):
    limit = min(max(limit, 1), 1000)
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params = []
    if flagged is not None:
        query += " AND flagged = ?"
        params.append(int(flagged))
    if project_name:
        query += " AND project_name = ?"
        params.append(project_name)
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return {"count": 0, "results": []}

    df["risk_labels"] = df["risk_labels"].apply(parse_json_list)
    df["tags"] = df["tags"].apply(parse_json_list)
    df["pii_types"] = df["pii_types"].apply(parse_json_list)

    return {"count": len(df), "results": df.to_dict(orient="records")}


@app.get("/export")
async def export_logs():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM audit_logs", conn)
    finally:
        conn.close()
    csv_data = df.to_csv(index=False)
    return Response(content=csv_data, media_type="text/csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
