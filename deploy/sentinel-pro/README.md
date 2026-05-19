# Sentinel-Pro Helm Chart

Install the full stack into a Kubernetes namespace:

```bash
helm install sentinel-pro ./deploy/sentinel-pro \
  --set image.repository=your-registry/sentinel-pro \
  --set web.image.repository=your-registry/sentinel-pro-web \
  --set auth.jwtSecret="$(openssl rand -hex 32)" \
  --set-string auth.oauthClients='admin-cli:replace-admin:admin,analyst-ui:replace-analyst:analyst,ingest-pipeline:replace-ingest:ingest'
```

The chart deploys the API, Celery worker, React/nginx web UI, Streamlit dashboard,
Postgres, and Redis by default. Set `postgres.enabled=false` with
`externalDatabaseUrl`, or `redis.enabled=false` with `externalRedisUrl`, to use managed
services instead.

Exchange client credentials at `/oauth/token` and use the returned JWT as
`Authorization: Bearer <token>`.
