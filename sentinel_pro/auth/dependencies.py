from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs

from fastapi import HTTPException, Request, Security
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field


class AuthRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    INGEST = "ingest"


VALID_ROLES = {role.value for role in AuthRole}
ROLE_SCOPES: Dict[str, Sequence[str]] = {
    AuthRole.ADMIN.value: ("sentinel:read", "sentinel:write", "sentinel:admin"),
    AuthRole.ANALYST.value: ("sentinel:read", "sentinel:write"),
    AuthRole.INGEST.value: ("sentinel:write",),
}

JWT_ALGORITHM = "HS256"
DEFAULT_JWT_ISSUER = "sentinel-pro"
DEFAULT_TOKEN_TTL_SECONDS = 3600
MIN_JWT_SECRET_LENGTH = 32

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/oauth/token",
    scopes={
        "sentinel:read": "Read audits, metrics, metadata, and incident reports.",
        "sentinel:write": "Create synchronous, batch, async, or webhook audits.",
        "sentinel:admin": "Access legacy logs and exports.",
    },
    auto_error=False,
)


class AuthContext(BaseModel):
    subject: str
    role: str
    scopes: List[str] = Field(default_factory=list)
    token_id: Optional[str] = None
    issuer: str = DEFAULT_JWT_ISSUER
    key: str = ""


class OAuthClient(BaseModel):
    client_id: str
    secret: str
    role: str


class OAuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str
    role: str


def _auth_header(error: str = "invalid_token") -> Dict[str, str]:
    return {"WWW-Authenticate": f'Bearer error="{error}"'}


def _raise_unauthorized(detail: str, error: str = "invalid_token") -> None:
    raise HTTPException(
        status_code=401,
        detail=detail,
        headers=_auth_header(error),
    )


def _normalize_role(role: str) -> str:
    value = role.strip().lower()
    if value not in VALID_ROLES:
        raise ValueError(f"Unknown auth role: {role}")
    return value


def _allowed_role_set(allowed_roles: Iterable[str]) -> set[str]:
    try:
        return {_normalize_role(role) for role in allowed_roles}
    except ValueError as exc:
        raise RuntimeError("Route configured with an unknown auth role") from exc


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))


def _json_segment(value: str) -> Dict[str, Any]:
    try:
        decoded = _base64url_decode(value)
        parsed = json.loads(decoded.decode("utf-8"))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JWT encoding") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Invalid JWT payload")
    return parsed


def _jwt_secret() -> str:
    secret = os.getenv("SENTINEL_JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("SENTINEL_JWT_SECRET is required for OAuth2 JWT auth")
    if len(secret) < MIN_JWT_SECRET_LENGTH:
        raise RuntimeError(
            f"SENTINEL_JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters"
        )
    return secret


def _jwt_issuer() -> str:
    return os.getenv("SENTINEL_JWT_ISSUER", DEFAULT_JWT_ISSUER).strip()


def _jwt_audience() -> Optional[str]:
    value = os.getenv("SENTINEL_JWT_AUDIENCE", "").strip()
    return value or None


def _jwt_leeway_seconds() -> int:
    return max(0, int(os.getenv("SENTINEL_JWT_LEEWAY_SEC", "30")))


def get_token_ttl_seconds() -> int:
    return max(
        60,
        int(
            os.getenv(
                "SENTINEL_JWT_ACCESS_TOKEN_TTL_SEC", str(DEFAULT_TOKEN_TTL_SECONDS)
            )
        ),
    )


def _sign(signing_input: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _base64url_encode(digest)


def encode_jwt(claims: Mapping[str, Any], secret: Optional[str] = None) -> str:
    signing_secret = secret or _jwt_secret()
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    encoded_header = _base64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    encoded_payload = _base64url_encode(
        json.dumps(dict(claims), separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{encoded_header}.{encoded_payload}"
    signature = _sign(signing_input, signing_secret)
    return f"{signing_input}.{signature}"


def create_access_token(
    subject: str,
    role: str,
    scopes: Optional[Sequence[str]] = None,
    ttl_seconds: Optional[int] = None,
) -> OAuthTokenResponse:
    normalized_role = _normalize_role(role)
    token_ttl = ttl_seconds or get_token_ttl_seconds()
    now = int(time.time())
    granted_scopes = list(scopes or ROLE_SCOPES[normalized_role])
    claims: Dict[str, Any] = {
        "sub": subject,
        "role": normalized_role,
        "scope": " ".join(granted_scopes),
        "iss": _jwt_issuer(),
        "iat": now,
        "nbf": now,
        "exp": now + token_ttl,
        "jti": secrets.token_urlsafe(24),
    }
    audience = _jwt_audience()
    if audience:
        claims["aud"] = audience
    return OAuthTokenResponse(
        access_token=encode_jwt(claims),
        expires_in=token_ttl,
        scope=" ".join(granted_scopes),
        role=normalized_role,
    )


def decode_access_token(token: str) -> AuthContext:
    parts = token.split(".")
    if len(parts) != 3:
        _raise_unauthorized("Invalid bearer token")

    encoded_header, encoded_payload, signature = parts
    try:
        header = _json_segment(encoded_header)
        payload = _json_segment(encoded_payload)
    except ValueError:
        _raise_unauthorized("Invalid bearer token")

    if header.get("alg") != JWT_ALGORITHM:
        _raise_unauthorized("Unsupported JWT algorithm")

    try:
        expected_signature = _sign(f"{encoded_header}.{encoded_payload}", _jwt_secret())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not hmac.compare_digest(signature, expected_signature):
        _raise_unauthorized("Invalid bearer token")

    now = int(time.time())
    leeway = _jwt_leeway_seconds()
    try:
        expires_at = int(payload["exp"])
    except (KeyError, TypeError, ValueError):
        _raise_unauthorized("Token is missing an expiration")
    if expires_at < now - leeway:
        _raise_unauthorized("Token has expired", "invalid_token")

    not_before = payload.get("nbf")
    if not_before is not None:
        try:
            not_before_at = int(not_before)
        except (TypeError, ValueError):
            _raise_unauthorized("Token has an invalid nbf claim")
        if not_before_at > now + leeway:
            _raise_unauthorized("Token is not valid yet")

    issuer = str(payload.get("iss") or "")
    if issuer != _jwt_issuer():
        _raise_unauthorized("Token issuer is invalid")

    audience = _jwt_audience()
    if audience:
        token_audience = payload.get("aud")
        if isinstance(token_audience, list):
            audience_valid = audience in [str(item) for item in token_audience]
        else:
            audience_valid = str(token_audience or "") == audience
        if not audience_valid:
            _raise_unauthorized("Token audience is invalid")

    subject = str(payload.get("sub") or "")
    if not subject:
        _raise_unauthorized("Token subject is missing")

    try:
        role = _normalize_role(str(payload.get("role") or ""))
    except ValueError:
        _raise_unauthorized("Token role is invalid")

    scope_claim = payload.get("scope", "")
    if isinstance(scope_claim, str):
        scopes = [scope for scope in scope_claim.split() if scope]
    elif isinstance(scope_claim, list):
        scopes = [str(scope) for scope in scope_claim if scope]
    else:
        scopes = []

    return AuthContext(
        subject=subject,
        role=role,
        scopes=scopes,
        token_id=str(payload.get("jti") or ""),
        issuer=issuer,
        key=str(payload.get("jti") or subject),
    )


def hash_oauth_secret(secret: str, iterations: int = 260_000) -> str:
    salt = secrets.token_urlsafe(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt}${_base64url_encode(digest)}"


def _verify_pbkdf2_secret(provided: str, stored: str) -> bool:
    try:
        _, iterations, salt, expected = stored.split("$", 3)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            provided.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(_base64url_encode(digest), expected)


def verify_oauth_secret(provided: str, stored: str) -> bool:
    if stored.startswith("pbkdf2_sha256$"):
        return _verify_pbkdf2_secret(provided, stored)
    return hmac.compare_digest(provided, stored)


def _client_from_mapping(client_id: str, value: Any) -> OAuthClient:
    if isinstance(value, str):
        try:
            secret, role = value.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid OAuth client entry for {client_id}") from exc
    elif isinstance(value, Mapping):
        secret = str(value.get("secret") or "")
        role = str(value.get("role") or "")
    else:
        raise ValueError(f"Invalid OAuth client entry for {client_id}")
    return OAuthClient(
        client_id=client_id,
        secret=secret,
        role=_normalize_role(role),
    )


def _load_json_clients(raw: str) -> Optional[Dict[str, OAuthClient]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    clients: Dict[str, OAuthClient] = {}
    if isinstance(parsed, Mapping):
        for client_id, value in parsed.items():
            clients[str(client_id)] = _client_from_mapping(str(client_id), value)
        return clients

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, Mapping):
                raise ValueError("OAuth client list entries must be objects")
            client_id = str(item.get("client_id") or "")
            if not client_id:
                raise ValueError("OAuth client entry is missing client_id")
            clients[client_id] = _client_from_mapping(client_id, item)
        return clients

    raise ValueError("SENTINEL_OAUTH_CLIENTS must be a JSON object or list")


def _load_delimited_clients(raw: str) -> Dict[str, OAuthClient]:
    clients: Dict[str, OAuthClient] = {}
    for item in [segment.strip() for segment in raw.split(",") if segment.strip()]:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "SENTINEL_OAUTH_CLIENTS entries must be client_id:client_secret:role"
            )
        client_id, secret, role = [part.strip() for part in parts]
        if not client_id or not secret:
            raise ValueError("OAuth client id and secret must be non-empty")
        clients[client_id] = OAuthClient(
            client_id=client_id,
            secret=secret,
            role=_normalize_role(role),
        )
    return clients


def load_oauth_clients() -> Dict[str, OAuthClient]:
    raw = os.getenv("SENTINEL_OAUTH_CLIENTS", "").strip()
    if raw:
        clients = _load_json_clients(raw)
        if clients is not None:
            return clients
        return _load_delimited_clients(raw)
    return {}


def _auth_required() -> bool:
    if os.getenv("SENTINEL_AUTH_DISABLED", "0") == "1":
        return False
    required = os.getenv("SENTINEL_AUTH_REQUIRED")
    if required is not None:
        return required == "1"
    return bool(load_oauth_clients())


def extract_key_from_headers(
    x_api_key: Optional[str], authorization: Optional[str]
) -> Optional[str]:
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


def _parse_requested_scopes(scope: str) -> List[str]:
    return [item for item in scope.split() if item]


def _authorize_requested_scopes(
    role: str, requested_scopes: Sequence[str]
) -> List[str]:
    role_scopes = set(ROLE_SCOPES[role])
    if not requested_scopes:
        return list(ROLE_SCOPES[role])

    requested = set(requested_scopes)
    if not requested.issubset(role_scopes):
        raise HTTPException(status_code=403, detail="Requested scope is not allowed")
    return [scope for scope in ROLE_SCOPES[role] if scope in requested]


def _authenticate_client(
    client_id: str,
    client_secret: str,
    requested_scopes: Sequence[str],
) -> OAuthTokenResponse:
    clients = load_oauth_clients()
    if not clients:
        raise HTTPException(status_code=500, detail="No OAuth clients are configured")

    client = clients.get(client_id)
    if client is None or not verify_oauth_secret(client_secret, client.secret):
        raise HTTPException(
            status_code=401,
            detail="Invalid OAuth client credentials",
            headers=_auth_header("invalid_client"),
        )

    scopes = _authorize_requested_scopes(client.role, requested_scopes)
    try:
        return create_access_token(
            subject=client.client_id,
            role=client.role,
            scopes=scopes,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def issue_oauth_token(request: Request) -> OAuthTokenResponse:
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type == "application/json":
        body = await request.json()
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Invalid token request body")
        values = {str(key): str(value) for key, value in body.items()}
    else:
        raw_body = (await request.body()).decode("utf-8")
        parsed = parse_qs(raw_body, keep_blank_values=True)
        values = {key: item[-1] for key, item in parsed.items() if item}

    grant_type = values.get("grant_type") or "client_credentials"
    scope = values.get("scope") or ""
    if grant_type == "client_credentials":
        client_id = values.get("client_id") or values.get("username") or ""
        client_secret = values.get("client_secret") or values.get("password") or ""
    elif grant_type == "password":
        client_id = values.get("username") or values.get("client_id") or ""
        client_secret = values.get("password") or values.get("client_secret") or ""
    else:
        raise HTTPException(status_code=400, detail="Unsupported OAuth2 grant type")

    if not client_id or not client_secret:
        raise HTTPException(status_code=400, detail="OAuth client credentials required")
    return _authenticate_client(
        client_id=client_id,
        client_secret=client_secret,
        requested_scopes=_parse_requested_scopes(scope),
    )


def require_roles(allowed_roles: Iterable[str]):
    allowed = _allowed_role_set(allowed_roles)

    def _dependency(token: Optional[str] = Security(oauth2_scheme)) -> AuthContext:
        if not _auth_required():
            return AuthContext(
                subject="auth-disabled",
                role=AuthRole.ADMIN.value,
                scopes=list(ROLE_SCOPES[AuthRole.ADMIN.value]),
                issuer=_jwt_issuer(),
                key="auth-disabled",
            )
        if not token:
            _raise_unauthorized("Missing bearer token")

        context = decode_access_token(token)
        if context.role not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return context

    return _dependency
