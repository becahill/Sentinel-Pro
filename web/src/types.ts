export type AuditRecord = {
  id: number;
  timestamp: string;
  input_text: string;
  output_text: string;
  toxicity_score: number;
  has_pii: boolean;
  is_refusal: boolean;
  self_harm: boolean;
  jailbreak: boolean;
  bias: boolean;
  sentiment_score: number;
  risk_labels: string[];
  risk_explanations: string[];
  pii_types: string[];
  flagged: boolean;
  redaction_applied: boolean;
  redaction_count: number;
  project_name?: string | null;
  model_name?: string | null;
  user_id?: string | null;
  request_id?: string | null;
  tags: string[];
};

export type Metrics = {
  total: number;
  flagged: number;
  flagged_rate: number;
  avg_toxicity: number;
  refusal_rate: number;
  self_harm_rate: number;
  jailbreak_rate: number;
  bias_rate: number;
  pii_rate: number;
  latest_timestamp: string | null;
  recent_24h: number;
  risk_counts: Record<string, number>;
  runtime: RuntimeMetrics;
};

export type RuntimeMetrics = {
  request_count: number;
  error_count: number;
  error_rate: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  status_codes: Record<string, number>;
  uptime_seconds: number;
  queue: {
    depth: number;
    queued: number;
    processing: number;
    completed: number;
    failed: number;
  };
};

export type Meta = {
  projects: string[];
  models: string[];
  users: string[];
  tags: string[];
  risk_labels: string[];
};

export type AuditListResponse = {
  page: number;
  page_size: number;
  total: number;
  results: AuditRecord[];
};

export type AuditCreatePayload = {
  input_text: string;
  output_text: string;
  project_name?: string;
  model_name?: string;
  user_id?: string;
  request_id?: string;
  tags?: string[];
  timestamp?: string;
};

export type IncidentReportJson = {
  generated_at: string;
  count: number;
  filters: Record<string, string | number | boolean>;
  records: AuditRecord[];
};
