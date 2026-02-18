# Security Guide

## Authentication and RBAC

Sentinel-Pro uses API keys mapped to roles via `SENTINEL_API_KEYS`.

Format:

```bash
SENTINEL_API_KEYS=admin:prod-admin,analyst:prod-analyst,ingest:prod-ingest
```

Accepted headers:

- `Authorization: Bearer <key>`
- `X-API-Key: <key>`

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

- Issue distinct keys per service/user and role.
- Rotate keys on a schedule and immediately on suspected exposure.
- Never commit keys to git or frontend bundles.
- Prefer environment injection from your secret manager.

## Rate limiting

The API enforces fixed-window rate limits per API key (or client IP when unauthenticated).

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
