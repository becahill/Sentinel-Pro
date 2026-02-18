import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import {
  createAudit,
  exportIncidentReport,
  getMeta,
  getMetrics,
  listAudits,
  setApiKey
} from "./api";
import type { AuditCreatePayload, AuditRecord, Meta, Metrics } from "./types";

const riskLabels: Record<string, { label: string; color: string }> = {
  toxicity: { label: "Toxicity", color: "var(--risk-toxic)" },
  pii: { label: "PII", color: "var(--risk-pii)" },
  self_harm: { label: "Self-harm", color: "var(--risk-self)" },
  jailbreak: { label: "Jailbreak", color: "var(--risk-jailbreak)" },
  bias: { label: "Bias", color: "var(--risk-bias)" },
  refusal: { label: "Refusal", color: "var(--risk-refusal)" }
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(handle);
  }, [value, delay]);
  return debounced;
}

function downloadTextFile(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [audits, setAudits] = useState<AuditRecord[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [selected, setSelected] = useState<AuditRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [total, setTotal] = useState(0);
  const [reloadToken, setReloadToken] = useState(0);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportStatus, setReportStatus] = useState<string | null>(null);
  const [apiKey, setApiKeyState] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }
    return localStorage.getItem("sentinel_api_key") || "";
  });

  const [filters, setFilters] = useState({
    search: "",
    flagged: "all",
    project_name: "",
    model_name: "",
    user_id: "",
    tag: "",
    risk_label: "",
    start_date: "",
    end_date: ""
  });

  const debouncedSearch = useDebounced(filters.search, 350);

  const totalPages = useMemo(() => {
    return Math.max(1, Math.ceil(total / pageSize));
  }, [total, pageSize]);

  const incidentTimeline = useMemo(() => {
    return audits
      .filter((record) => record.flagged)
      .slice(0, 10)
      .map((record) => ({
        id: record.id,
        timestamp: formatTimestamp(record.timestamp),
        project: record.project_name || "-",
        labels: record.risk_labels.map((label) => riskLabels[label]?.label ?? label),
        explanation: record.risk_explanations[0] || "No explanation provided"
      }));
  }, [audits]);

  useEffect(() => {
    setApiKey(apiKey);
    if (typeof window !== "undefined") {
      if (apiKey) {
        localStorage.setItem("sentinel_api_key", apiKey);
      } else {
        localStorage.removeItem("sentinel_api_key");
      }
    }
  }, [apiKey]);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const response = await getMeta();
        setMeta(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load metadata");
      }
    };

    const loadMetrics = async () => {
      try {
        const response = await getMetrics();
        setMetrics(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load metrics");
      }
    };

    loadMeta();
    loadMetrics();
  }, []);

  useEffect(() => {
    const loadAudits = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await listAudits({
          page,
          page_size: pageSize,
          search: debouncedSearch,
          flagged:
            filters.flagged === "all"
              ? undefined
              : filters.flagged === "flagged",
          project_name: filters.project_name || undefined,
          model_name: filters.model_name || undefined,
          user_id: filters.user_id || undefined,
          tag: filters.tag || undefined,
          risk_label: filters.risk_label || undefined,
          start_date: filters.start_date || undefined,
          end_date: filters.end_date || undefined
        });
        setAudits(response.results);
        setTotal(response.total);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load audits");
      } finally {
        setLoading(false);
      }
    };

    loadAudits();
  }, [
    page,
    pageSize,
    filters.flagged,
    filters.project_name,
    filters.model_name,
    filters.user_id,
    filters.tag,
    filters.risk_label,
    filters.start_date,
    filters.end_date,
    debouncedSearch,
    reloadToken
  ]);

  useEffect(() => {
    setPage(1);
  }, [
    filters.flagged,
    filters.project_name,
    filters.model_name,
    filters.user_id,
    filters.tag,
    filters.risk_label,
    filters.start_date,
    filters.end_date,
    debouncedSearch
  ]);

  const refreshMetrics = async () => {
    try {
      const response = await getMetrics();
      setMetrics(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh metrics");
    }
  };

  const handleSubmit = async (payload: AuditCreatePayload, disableToxicity: boolean) => {
    try {
      setError(null);
      await createAudit(payload, disableToxicity);
      setPage(1);
      setReloadToken((value) => value + 1);
      await refreshMetrics();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create audit");
    }
  };

  const handleExportIncidentReport = async () => {
    setReportLoading(true);
    setReportStatus(null);
    try {
      const report = await exportIncidentReport({
        output_format: "markdown",
        flagged_only: filters.flagged === "safe" ? false : true,
        project_name: filters.project_name || undefined,
        model_name: filters.model_name || undefined,
        user_id: filters.user_id || undefined,
        risk_label: filters.risk_label || undefined,
        start_date: filters.start_date || undefined,
        end_date: filters.end_date || undefined,
        limit: 400
      });
      downloadTextFile(report.filename, report.content, "text/markdown;charset=utf-8");
      setReportStatus(`Exported ${report.filename}`);
    } catch (err) {
      setReportStatus(err instanceof Error ? err.message : "Failed to export report");
    } finally {
      setReportLoading(false);
    }
  };

  const applyFlaggedFocus = (risk?: string) => {
    setFilters((prev) => ({
      ...prev,
      flagged: "flagged",
      risk_label: risk || ""
    }));
  };

  const clearFilters = () => {
    setFilters({
      search: "",
      flagged: "all",
      project_name: "",
      model_name: "",
      user_id: "",
      tag: "",
      risk_label: "",
      start_date: "",
      end_date: ""
    });
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="brand">
            <div className="logo">SP</div>
            <div>
              <div className="brand-title">Sentinel-Pro</div>
              <div className="brand-subtitle">
                Safety audit control panel for LLM output monitoring
              </div>
            </div>
          </div>
          <div className="header-actions">
            <div className="api-key">
              <label>API Key</label>
              <input
                type="password"
                value={apiKey}
                onChange={(event) => setApiKeyState(event.target.value)}
                placeholder="Paste key"
              />
            </div>
            <div className="status-chip">Live</div>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <h1>Operationalize safety audits in real time.</h1>
            <p>
              Submit model outputs, inspect risk signals, and triage incidents from a
              single, production-grade workspace.
            </p>
            <div className="quick-actions">
              <button className="ghost" onClick={() => applyFlaggedFocus()}>
                Flagged Only
              </button>
              <button className="ghost" onClick={() => applyFlaggedFocus("pii")}>
                PII Cases
              </button>
              <button
                className="ghost"
                onClick={() => applyFlaggedFocus("jailbreak")}
              >
                Jailbreak Cases
              </button>
              <button className="ghost" onClick={clearFilters}>
                Clear Filters
              </button>
            </div>
          </div>
          <div className="hero-card">
            <div className="hero-label">Latest ingest</div>
            <div className="hero-value">
              {metrics?.latest_timestamp
                ? formatTimestamp(metrics.latest_timestamp)
                : "No data"}
            </div>
            <div className="hero-meta">
              {metrics ? `${metrics.recent_24h} audits in the last 24h` : ""}
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="card span-7" data-delay="1">
            <div className="card-title">Quick Audit</div>
            <AuditForm onSubmit={handleSubmit} />
          </div>
          <div className="card span-5" data-delay="2">
            <div className="card-title">Risk Overview</div>
            <div className="stats">
              <Stat label="Total Audits" value={metrics?.total ?? 0} />
              <Stat label="Flagged" value={metrics?.flagged ?? 0} tone="danger" />
              <Stat
                label="Flagged Rate"
                value={metrics ? formatPercent(metrics.flagged_rate) : "-"}
              />
              <Stat
                label="Avg Toxicity"
                value={metrics ? metrics.avg_toxicity.toFixed(3) : "-"}
              />
              <Stat
                label="PII Rate"
                value={metrics ? formatPercent(metrics.pii_rate) : "-"}
              />
              <Stat
                label="Refusal Rate"
                value={metrics ? formatPercent(metrics.refusal_rate) : "-"}
              />
            </div>
            <div className="risk-bars">
              {metrics &&
                Object.entries(riskLabels).map(([key, info]) => (
                  <RiskBar
                    key={key}
                    label={info.label}
                    color={info.color}
                    count={metrics.risk_counts[key] ?? 0}
                    max={Math.max(1, ...Object.values(metrics.risk_counts || { 1: 1 }))}
                  />
                ))}
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="card span-6" data-delay="2">
            <div className="card-title">Runtime Telemetry</div>
            <div className="stats">
              <Stat
                label="P50 Latency"
                value={
                  metrics ? `${metrics.runtime.latency_ms_p50.toFixed(1)} ms` : "-"
                }
              />
              <Stat
                label="P95 Latency"
                value={
                  metrics ? `${metrics.runtime.latency_ms_p95.toFixed(1)} ms` : "-"
                }
              />
              <Stat
                label="Error Rate"
                value={metrics ? formatPercent(metrics.runtime.error_rate) : "-"}
                tone={metrics && metrics.runtime.error_rate > 0 ? "danger" : undefined}
              />
              <Stat
                label="Queue Depth"
                value={metrics ? metrics.runtime.queue.depth : "-"}
              />
              <Stat
                label="Queue Failed"
                value={metrics ? metrics.runtime.queue.failed : "-"}
                tone={metrics && metrics.runtime.queue.failed > 0 ? "danger" : undefined}
              />
              <Stat
                label="Requests"
                value={metrics ? metrics.runtime.request_count : "-"}
              />
            </div>
            {metrics && (
              <div className="status-code-grid">
                {Object.entries(metrics.runtime.status_codes).map(([status, count]) => (
                  <div key={status} className="status-code-pill">
                    <span className="mono">{status}</span>
                    <span>{count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="card span-6" data-delay="3">
            <div className="card-title">Incident Timeline</div>
            <div className="timeline">
              {incidentTimeline.length === 0 && (
                <div className="muted">No flagged incidents in this page of results.</div>
              )}
              {incidentTimeline.map((incident) => (
                <button
                  key={incident.id}
                  className="timeline-item"
                  onClick={() => {
                    const match = audits.find((record) => record.id === incident.id);
                    if (match) {
                      setSelected(match);
                    }
                  }}
                >
                  <div className="timeline-header">
                    <span className="mono">#{incident.id}</span>
                    <span>{incident.timestamp}</span>
                  </div>
                  <div className="timeline-project">{incident.project}</div>
                  <div className="timeline-labels">{incident.labels.join(", ")}</div>
                  <div className="timeline-explanation">{incident.explanation}</div>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="card span-12" data-delay="3">
            <div className="section-header">
              <div className="card-title">Audit Stream</div>
              <button
                className="primary"
                onClick={handleExportIncidentReport}
                disabled={reportLoading}
              >
                {reportLoading ? "Exporting..." : "Export Incident Report"}
              </button>
            </div>
            {reportStatus && <div className="banner success">{reportStatus}</div>}

            <div className="filters">
              <div className="field">
                <label>Search</label>
                <input
                  value={filters.search}
                  onChange={(event) =>
                    setFilters({ ...filters, search: event.target.value })
                  }
                  placeholder="Search input or output text"
                />
              </div>
              <div className="field">
                <label>Status</label>
                <select
                  value={filters.flagged}
                  onChange={(event) =>
                    setFilters({ ...filters, flagged: event.target.value })
                  }
                >
                  <option value="all">All</option>
                  <option value="flagged">Flagged</option>
                  <option value="safe">Safe</option>
                </select>
              </div>
              <div className="field">
                <label>Project</label>
                <select
                  value={filters.project_name}
                  onChange={(event) =>
                    setFilters({ ...filters, project_name: event.target.value })
                  }
                >
                  <option value="">All</option>
                  {meta?.projects.map((project) => (
                    <option key={project} value={project}>
                      {project}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Model</label>
                <select
                  value={filters.model_name}
                  onChange={(event) =>
                    setFilters({ ...filters, model_name: event.target.value })
                  }
                >
                  <option value="">All</option>
                  {meta?.models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>User</label>
                <select
                  value={filters.user_id}
                  onChange={(event) =>
                    setFilters({ ...filters, user_id: event.target.value })
                  }
                >
                  <option value="">All</option>
                  {meta?.users.map((user) => (
                    <option key={user} value={user}>
                      {user}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Tag</label>
                <select
                  value={filters.tag}
                  onChange={(event) =>
                    setFilters({ ...filters, tag: event.target.value })
                  }
                >
                  <option value="">All</option>
                  {meta?.tags.map((tag) => (
                    <option key={tag} value={tag}>
                      {tag}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Risk Label</label>
                <select
                  value={filters.risk_label}
                  onChange={(event) =>
                    setFilters({ ...filters, risk_label: event.target.value })
                  }
                >
                  <option value="">All</option>
                  {meta?.risk_labels.map((label) => (
                    <option key={label} value={label}>
                      {riskLabels[label]?.label ?? label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Start date</label>
                <input
                  type="date"
                  value={filters.start_date}
                  onChange={(event) =>
                    setFilters({ ...filters, start_date: event.target.value })
                  }
                />
              </div>
              <div className="field">
                <label>End date</label>
                <input
                  type="date"
                  value={filters.end_date}
                  onChange={(event) =>
                    setFilters({ ...filters, end_date: event.target.value })
                  }
                />
              </div>
            </div>

            {error && <div className="banner error">{error}</div>}

            <div className="table">
              <div className="table-head">
                <span>Status</span>
                <span>Timestamp</span>
                <span>Project</span>
                <span>Risk</span>
                <span>Input</span>
                <span>Output</span>
              </div>
              {loading && <div className="table-row">Loading audits...</div>}
              {!loading && audits.length === 0 && (
                <div className="table-row">No audits found.</div>
              )}
              {audits.map((record) => (
                <button
                  key={record.id}
                  className="table-row clickable"
                  onClick={() => setSelected(record)}
                >
                  <div>
                    <span
                      className={`status-pill ${record.flagged ? "danger" : "safe"}`}
                    >
                      {record.flagged ? "Flagged" : "Safe"}
                    </span>
                  </div>
                  <div className="mono">{formatTimestamp(record.timestamp)}</div>
                  <div>{record.project_name || "-"}</div>
                  <div className="risk-list">
                    {record.risk_labels.length === 0 && (
                      <span className="risk-pill neutral">None</span>
                    )}
                    {record.risk_labels.map((label) => (
                      <span
                        key={label}
                        className="risk-pill"
                        style={{
                          borderColor: riskLabels[label]?.color
                        }}
                      >
                        {riskLabels[label]?.label ?? label}
                      </span>
                    ))}
                  </div>
                  <div className="truncate">{record.input_text}</div>
                  <div className="truncate">{record.output_text}</div>
                </button>
              ))}
            </div>

            <div className="pagination">
              <span>
                Page {page} of {totalPages}
              </span>
              <div className="pager-controls">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                >
                  Previous
                </button>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
                >
                  Next
                </button>
                <button onClick={() => setReloadToken((value) => value + 1)}>Refresh</button>
              </div>
            </div>
          </div>
        </section>

        <footer className="footer">
          <span>Sentinel-Pro Safety Audit Suite</span>
          <span className="mono">API: /api</span>
        </footer>
      </div>

      {selected && (
        <div className="drawer-overlay" onClick={() => setSelected(null)}>
          <aside className="drawer" onClick={(event) => event.stopPropagation()}>
            <div className="drawer-header">
              <div>
                <div className="drawer-title">Audit Record #{selected.id}</div>
                <div className="drawer-subtitle">
                  {formatTimestamp(selected.timestamp)}
                </div>
              </div>
              <button className="ghost" onClick={() => setSelected(null)}>
                Close
              </button>
            </div>
            <div className="drawer-section">
              <div className="detail-grid">
                <Detail label="Project" value={selected.project_name || "-"} />
                <Detail label="Model" value={selected.model_name || "-"} />
                <Detail label="User" value={selected.user_id || "-"} />
                <Detail label="Request ID" value={selected.request_id || "-"} />
                <Detail label="Toxicity" value={selected.toxicity_score.toFixed(3)} />
                <Detail
                  label="Sentiment"
                  value={selected.sentiment_score.toFixed(3)}
                />
                <Detail label="PII" value={selected.has_pii ? "Yes" : "No"} />
                <Detail
                  label="Redactions"
                  value={
                    selected.redaction_applied ? `${selected.redaction_count}` : "None"
                  }
                />
              </div>
            </div>
            <div className="drawer-section">
              <div className="drawer-label">Risk Explanations</div>
              {selected.risk_explanations.length === 0 && (
                <div className="muted">No explanations recorded.</div>
              )}
              <ul className="explanations">
                {selected.risk_explanations.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
            <div className="drawer-section">
              <div className="drawer-label">Input</div>
              <div className="drawer-text">{selected.input_text}</div>
            </div>
            <div className="drawer-section">
              <div className="drawer-label">Output</div>
              <div className="drawer-text">{selected.output_text}</div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  tone
}: {
  label: string;
  value: string | number;
  tone?: "danger";
}) {
  return (
    <div className={`stat ${tone || ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function RiskBar({
  label,
  color,
  count,
  max
}: {
  label: string;
  color: string;
  count: number;
  max: number;
}) {
  const width = Math.max(8, Math.round((count / max) * 100));
  return (
    <div className="risk-bar">
      <span>{label}</span>
      <div className="risk-track">
        <div className="risk-fill" style={{ width: `${width}%`, background: color }} />
      </div>
      <span className="mono">{count}</span>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="detail-label">{label}</div>
      <div className="detail-value">{value}</div>
    </div>
  );
}

function AuditForm({
  onSubmit
}: {
  onSubmit: (payload: AuditCreatePayload, disableToxicity: boolean) => void;
}) {
  const [form, setForm] = useState({
    input_text: "",
    output_text: "",
    project_name: "",
    model_name: "",
    user_id: "",
    request_id: "",
    tags: "",
    timestamp: "",
    disable_toxicity: false
  });
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (
    event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(null);
    if (!form.input_text.trim() || !form.output_text.trim()) {
      setError("Input and output text are required.");
      return;
    }

    setSubmitting(true);
    const payload: AuditCreatePayload = {
      input_text: form.input_text.trim(),
      output_text: form.output_text.trim()
    };

    if (form.project_name.trim()) {
      payload.project_name = form.project_name.trim();
    }
    if (form.model_name.trim()) {
      payload.model_name = form.model_name.trim();
    }
    if (form.user_id.trim()) {
      payload.user_id = form.user_id.trim();
    }
    if (form.request_id.trim()) {
      payload.request_id = form.request_id.trim();
    }
    if (form.tags.trim()) {
      payload.tags = form.tags
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
    }
    if (form.timestamp.trim()) {
      payload.timestamp = form.timestamp.trim();
    }

    try {
      await onSubmit(payload, form.disable_toxicity);
      setForm((prev) => ({
        ...prev,
        input_text: "",
        output_text: "",
        tags: "",
        request_id: ""
      }));
      setSuccess("Audit submitted successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Submission failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="form" onSubmit={handleSubmit}>
      <div className="field">
        <label>Input text</label>
        <textarea
          name="input_text"
          value={form.input_text}
          onChange={handleChange}
          placeholder="User prompt"
          rows={4}
        />
      </div>
      <div className="field">
        <label>Output text</label>
        <textarea
          name="output_text"
          value={form.output_text}
          onChange={handleChange}
          placeholder="Model response"
          rows={4}
        />
      </div>
      <div className="form-grid">
        <div className="field">
          <label>Project</label>
          <input
            name="project_name"
            value={form.project_name}
            onChange={handleChange}
            placeholder="demo"
          />
        </div>
        <div className="field">
          <label>Model</label>
          <input
            name="model_name"
            value={form.model_name}
            onChange={handleChange}
            placeholder="gpt-4o-mini"
          />
        </div>
        <div className="field">
          <label>User</label>
          <input
            name="user_id"
            value={form.user_id}
            onChange={handleChange}
            placeholder="user-001"
          />
        </div>
        <div className="field">
          <label>Request ID</label>
          <input
            name="request_id"
            value={form.request_id}
            onChange={handleChange}
            placeholder="req-123"
          />
        </div>
        <div className="field">
          <label>Tags</label>
          <input
            name="tags"
            value={form.tags}
            onChange={handleChange}
            placeholder="pii, jailbreak"
          />
        </div>
        <div className="field">
          <label>Timestamp</label>
          <input
            name="timestamp"
            value={form.timestamp}
            onChange={handleChange}
            placeholder="2026-02-02T12:00:00"
          />
        </div>
      </div>
      <div className="toggle-row">
        <label className="toggle">
          <input
            type="checkbox"
            checked={form.disable_toxicity}
            onChange={(event) =>
              setForm((prev) => ({
                ...prev,
                disable_toxicity: event.target.checked
              }))
            }
          />
          <span>Disable toxicity model</span>
        </label>
      </div>
      {error && <div className="banner error">{error}</div>}
      {success && <div className="banner success">{success}</div>}
      <button className="primary" type="submit" disabled={submitting}>
        {submitting ? "Submitting..." : "Run audit"}
      </button>
    </form>
  );
}
