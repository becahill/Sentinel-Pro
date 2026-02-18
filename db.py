from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

audit_logs = Table(
    "audit_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("input_text", Text, nullable=False),
    Column("output_text", Text, nullable=False),
    Column("toxicity_score", Float),
    Column("has_pii", Boolean),
    Column("is_refusal", Boolean),
    Column("self_harm", Boolean),
    Column("jailbreak", Boolean),
    Column("bias", Boolean),
    Column("sentiment_score", Float),
    Column("risk_labels", Text),
    Column("risk_explanations", Text),
    Column("pii_types", Text),
    Column("flagged", Boolean),
    Column("redaction_applied", Boolean),
    Column("redaction_count", Integer),
    Column("project_name", Text),
    Column("model_name", Text),
    Column("user_id", Text),
    Column("request_id", Text),
    Column("tags", Text),
)

Index("ix_audit_logs_timestamp", audit_logs.c.timestamp)
Index("ix_audit_logs_flagged", audit_logs.c.flagged)


def resolve_db_url(db_path: Optional[str] = None) -> str:
    if db_path:
        if "://" in db_path:
            return db_path
        return f"sqlite:///{db_path}"
    url = os.getenv("SENTINEL_DB_URL")
    if url:
        return url
    path = os.getenv("SENTINEL_DB_PATH", "audit_logs.db")
    return f"sqlite:///{path}"


def get_engine(db_path: Optional[str] = None) -> Engine:
    url = resolve_db_url(db_path)
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def init_db(engine: Engine, auto_create: Optional[bool] = None) -> None:
    if auto_create is None:
        auto_create = os.getenv("SENTINEL_AUTO_MIGRATE", "1") != "0"
    if auto_create:
        metadata.create_all(engine)
