import type {
  AuditCreatePayload,
  AuditListResponse,
  AuditRecord,
  IncidentReportJson,
  Meta,
  Metrics
} from "./types";

const RAW_API_URL = import.meta.env.VITE_API_URL;
const API_URL = RAW_API_URL === undefined ? "http://localhost:8000" : RAW_API_URL;
let apiKey = "";

type QueryValue = string | number | boolean | undefined | null;

interface QueryParams {
  [key: string]: QueryValue;
}

function buildQuery(params: QueryParams): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    search.append(key, String(value));
  });
  const query = search.toString();
  return query ? `?${query}` : "";
}

function buildUrl(path: string): string {
  if (!API_URL) {
    return path;
  }
  const trimmed = API_URL.endsWith("/") ? API_URL.slice(0, -1) : API_URL;
  return `${trimmed}${path}`;
}

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = buildHeaders(options);
  if (!apiKey && typeof window !== "undefined") {
    const stored = localStorage.getItem("sentinel_api_key");
    if (stored) {
      apiKey = stored;
    }
  }
  if (apiKey && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${apiKey}`);
  }
  const response = await fetch(buildUrl(path), {
    ...options,
    headers
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed");
  }
  return (await response.json()) as T;
}

function buildHeaders(options?: RequestInit): Headers {
  const headers = new Headers(options?.headers);
  if (options?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

export async function listAudits(params: QueryParams): Promise<AuditListResponse> {
  return fetchJson(`/api/audits${buildQuery(params)}`);
}

export async function getAudit(auditId: number): Promise<AuditRecord> {
  return fetchJson(`/api/audits/${auditId}`);
}

export async function getMetrics(): Promise<Metrics> {
  return fetchJson("/api/metrics");
}

export async function getMeta(): Promise<Meta> {
  return fetchJson("/api/meta");
}

export async function createAudit(
  payload: AuditCreatePayload,
  disableToxicity: boolean
): Promise<void> {
  const query = buildQuery({ disable_toxicity: disableToxicity });
  await fetchJson(`/api/audits${query}`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function exportIncidentReport(
  params: QueryParams
): Promise<{ filename: string; content: string }> {
  const headers = buildHeaders();
  if (!apiKey && typeof window !== "undefined") {
    const stored = localStorage.getItem("sentinel_api_key");
    if (stored) {
      apiKey = stored;
    }
  }
  if (apiKey && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${apiKey}`);
  }

  const response = await fetch(
    buildUrl(`/api/reports/incidents${buildQuery(params)}`),
    {
      method: "GET",
      headers
    }
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed");
  }
  const content = await response.text();
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
  return {
    filename: match?.[1] || "incident_report.md",
    content
  };
}

export async function getIncidentReportJson(
  params: QueryParams
): Promise<IncidentReportJson> {
  return fetchJson(
    `/api/reports/incidents${buildQuery({ ...params, output_format: "json" })}`
  );
}

export function setApiKey(key: string) {
  apiKey = key;
}
