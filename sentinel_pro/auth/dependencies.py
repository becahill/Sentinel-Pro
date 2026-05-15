from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

from fastapi import Header, HTTPException
from pydantic import BaseModel


class AuthContext(BaseModel):
    key: str
    role: str


def _load_api_keys() -> Dict[str, str]:
    raw = os.getenv("SENTINEL_API_KEYS", "")
    keys: Dict[str, str] = {}
    for item in [segment.strip() for segment in raw.split(",") if segment.strip()]:
        if ":" in item:
            role, key = item.split(":", 1)
            role = role.strip().lower()
            key = key.strip()
        else:
            key = item
            role = "analyst"
        if key:
            keys[key] = role
    return keys


def _auth_required() -> bool:
    if os.getenv("SENTINEL_AUTH_DISABLED", "0") == "1":
        return False
    required = os.getenv("SENTINEL_AUTH_REQUIRED")
    if required is not None:
        return required == "1"
    return bool(_load_api_keys())


def extract_key_from_headers(
    x_api_key: Optional[str], authorization: Optional[str]
) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


def require_roles(allowed_roles: Iterable[str]):
    allowed = {role.lower() for role in allowed_roles}

    def _dependency(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
    ) -> AuthContext:
        if not _auth_required():
            return AuthContext(key="", role="admin")
        api_keys = _load_api_keys()
        key = extract_key_from_headers(x_api_key, authorization)
        if not key:
            raise HTTPException(status_code=401, detail="Missing API key")
        role = api_keys.get(key)
        if not role:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if role.lower() not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return AuthContext(key=key, role=role)

    return _dependency
