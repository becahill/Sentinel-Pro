from __future__ import annotations

import os
from typing import Optional, Tuple

from celery import Celery

from sentinel_pro.storage.cache import get_redis_client


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def get_celery_queue_name() -> str:
    return os.getenv("SENTINEL_CELERY_QUEUE", "audits")


def get_celery_broker_url() -> str:
    return (
        os.getenv("SENTINEL_CELERY_BROKER_URL")
        or os.getenv("SENTINEL_REDIS_URL")
        or "redis://localhost:6379/0"
    )


def get_celery_result_backend() -> str:
    if is_celery_eager() and not os.getenv("SENTINEL_CELERY_RESULT_BACKEND"):
        return "cache+memory://"
    return (
        os.getenv("SENTINEL_CELERY_RESULT_BACKEND")
        or os.getenv("SENTINEL_REDIS_URL")
        or "redis://localhost:6379/1"
    )


def get_queue_result_ttl_seconds() -> int:
    return max(60, int(os.getenv("SENTINEL_QUEUE_RESULT_TTL_SEC", "3600")))


def is_celery_eager() -> bool:
    return env_bool("SENTINEL_CELERY_TASK_ALWAYS_EAGER", False)


celery_app = Celery(
    "sentinel_pro",
    broker=get_celery_broker_url(),
    backend=get_celery_result_backend(),
    include=["sentinel_pro.jobs.audits"],
)

celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    result_expires=get_queue_result_ttl_seconds(),
    result_serializer="json",
    task_acks_late=True,
    task_default_queue=get_celery_queue_name(),
    task_reject_on_worker_lost=True,
    task_serializer="json",
    task_store_eager_result=True,
    task_track_started=True,
    timezone="UTC",
    worker_prefetch_multiplier=max(
        1, int(os.getenv("SENTINEL_CELERY_PREFETCH_MULTIPLIER", "1"))
    ),
)

if is_celery_eager():
    celery_app.conf.task_always_eager = True


def check_celery_broker_ready() -> Tuple[bool, Optional[str]]:
    if is_celery_eager():
        return True, None

    broker_url = get_celery_broker_url()
    if broker_url.startswith("redis://") or broker_url.startswith("rediss://"):
        client = get_redis_client(broker_url)
        if client is None:
            return False, "Redis client is not available"
        try:
            client.ping()
        except Exception as exc:
            return False, str(exc)
        return True, None

    try:
        with celery_app.connection_for_read() as conn:
            conn.ensure_connection(max_retries=1)
    except Exception as exc:
        return False, str(exc)
    return True, None


def get_celery_queue_depth() -> Optional[int]:
    if is_celery_eager():
        return 0

    broker_url = get_celery_broker_url()
    if not broker_url.startswith(("redis://", "rediss://")):
        return None

    client = get_redis_client(broker_url)
    if client is None:
        return None
    try:
        return int(client.llen(get_celery_queue_name()))
    except Exception:
        return None
