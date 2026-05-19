# Security Guide

## Authentication and RBAC

Sentinel-Pro uses OAuth2 client credentials to issue short-lived signed JWT access
tokens. API routes accept only `Authorization: Bearer <jwt>`.

Configure OAuth2 clients and the JWT signing secret:

```bash
SENTINEL_OAUTH_CLIENTS=admin-cli:prod-admin-secret:admin,analyst-ui:prod-analyst-secret:analyst,ingest-pipeline:prod-ingest-secret:ingest
SENTINEL_JWT_SECRET=replace-with-at-least-32-random-characters
```

Request a token:

```bash
TOKEN=$(
  curl -s -X POST http://localhost:8000/oauth/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials&client_id=analyst-ui&client_secret=prod-analyst-secret" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
)
```

Use it on API requests:

```http
Authorization: Bearer <jwt>
```

Role model:

| Role | Intended use | Access |
| --- | --- | --- |
| `admin` | Operators and SREs | Full access, including `/logs` and `/export` |
| `analyst` | Safety reviewers | Read + audit creation endpoints |
| `ingest` | Automated pipelines | Write ingestion endpoints only |

Route policy:

- Read routes (`/api/audits`, `/api/metrics`, `/api/meta`, `/api/reports/incidents`): `admin`, `analyst`
- Write routes (`/api/audits`, `/api/audits/batch`, `/api/audits/async`, `/webhook`): `admin`, `analyst`, `ingest`
- Admin export/log routes (`/logs`, `/export`): `admin`

## Production defaults

Set these for production:

```bash
SENTINEL_AUTH_REQUIRED=1
SENTINEL_AUTH_DISABLED=0
SENTINEL_REDACT_PII=1
SENTINEL_RATE_LIMIT_REQUESTS=120
SENTINEL_RATE_LIMIT_WINDOW_SEC=60
```

## Key management recommendations

- Issue distinct OAuth2 clients per service/user and role.
- Store client secrets and `SENTINEL_JWT_SECRET` in a secret manager.
- Rotate client secrets and the JWT signing secret on a schedule and immediately on suspected exposure.
- Prefer PBKDF2-hashed client secrets in `SENTINEL_OAUTH_CLIENTS` for production.
- Prefer environment injection from your secret manager.

## Rate limiting

The API enforces fixed-window rate limits per bearer token (or client IP when
unauthenticated).

Headers returned on rate-limited routes:

- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

A limit breach returns `429 Rate limit exceeded`.

## Health/readiness and monitoring

- `GET /healthz`: process liveness
- `GET /readyz`: DB connectivity + queue worker readiness
- `GET /api/metrics`: runtime latency/error and queue telemetry

For full threat analysis, see `Sentinel-Pro-threat-model.md`.
