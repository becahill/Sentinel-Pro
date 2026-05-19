from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    import redis
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - exercised only when optional deps are absent
    redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]

METRICS_CACHE_PREFIX = "sentinel-pro:metrics:v1"

_REDIS_CLIENTS: Dict[str, Any] = {}
_REDIS_LOCK = threading.Lock()
_MEMORY_CACHE: Dict[str, Tuple[float, str]] = {}
_MEMORY_LOCK = threading.Lock()


def get_cache_redis_url() -> Optional[str]:
    return os.getenv("SENTINEL_CACHE_REDIS_URL") or os.getenv("SENTINEL_REDIS_URL")


def get_redis_client(url: Optional[str] = None) -> Optional[Any]:
    redis_url = url or get_cache_redis_url()
    if not redis_url or redis is None:
        return None

    with _REDIS_LOCK:
        client = _REDIS_CLIENTS.get(redis_url)
        if client is None:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            _REDIS_CLIENTS[redis_url] = client
        return client


def metrics_cache_key(db_url: str) -> str:
    digest = hashlib.sha256(db_url.encode("utf-8")).hexdigest()[:16]
    return f"{METRICS_CACHE_PREFIX}:{digest}"


def get_metrics_cache_ttl_seconds() -> int:
    return max(0, int(os.getenv("SENTINEL_METRICS_CACHE_TTL_SEC", "30")))


def cache_get_json(key: str) -> Optional[Dict[str, Any]]:
    client = get_redis_client()
    if client is not None:
        try:
            payload = client.get(key)
            if payload:
                return json.loads(payload)
        except (RedisError, TypeError, ValueError):
            pass

    now = time.time()
    with _MEMORY_LOCK:
        item = _MEMORY_CACHE.get(key)
        if not item:
            return None
        expires_at, payload = item
        if expires_at <= now:
            _MEMORY_CACHE.pop(key, None)
            return None
    try:
        return json.loads(payload)
    except ValueError:
        return None


def cache_set_json(key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return

    payload = json.dumps(value, default=str, separators=(",", ":"))
    client = get_redis_client()
    if client is not None:
        try:
            client.setex(key, ttl_seconds, payload)
            return
        except RedisError:
            pass

    with _MEMORY_LOCK:
        _MEMORY_CACHE[key] = (time.time() + ttl_seconds, payload)


def cache_delete(key: str, redis_url: Optional[str] = None) -> None:
    client = get_redis_client(redis_url)
    if client is not None:
        try:
            client.delete(key)
        except RedisError:
            pass

    with _MEMORY_LOCK:
        _MEMORY_CACHE.pop(key, None)


def invalidate_metrics_cache(db_url: str, redis_url: Optional[str] = None) -> None:
    cache_delete(metrics_cache_key(db_url), redis_url=redis_url)
