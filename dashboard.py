from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from auditor import AuditEngine, ConversationRecord, normalize_tags
from signals import SignalDetector

TOXICITY_THRESHOLD = 0.7

# Page Config
st.set_page_config(page_title="Sentinel-Pro Dashboard", layout="wide")

st.title("Sentinel-Pro: AI Safety Audit")


@st.cache_data(ttl=5)
def load_data(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM audit_logs", conn)
    finally:
        conn.close()
    return df


def parse_json_list(value) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [str(value)]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "project_name": "",
        "model_name": "",
        "user_id": "",
        "request_id": "",
        "tags": "[]",
        "risk_labels": "[]",
        "risk_explanations": "[]",
        "pii_types": "[]",
        "self_harm": False,
        "jailbreak": False,
        "bias": False,
        "redaction_applied": False,
        "redaction_count": 0,
    }
    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value
    return df


default_db_path = os.getenv("SENTINEL_DB_PATH", "audit_logs.db")

with st.sidebar:
    st.header("Data Source")
    db_path = st.text_input("SQLite DB path", default_db_path)

try:
    df = load_data(db_path)
except FileNotFoundError:
    st.error("Database not found. Run 'python auditor.py --demo' first.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load database: {exc}")
    st.stop()

if df.empty:
    st.warning("No data found. Run 'python auditor.py --demo' to generate logs.")
    st.stop()

df = ensure_columns(df)

for column in ["has_pii", "is_refusal", "flagged", "self_harm", "jailbreak", "bias"]:
    if column in df.columns:
        df[column] = df[column].astype(bool)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df["tags_list"] = df["tags"].apply(parse_json_list)

df["risk_labels_list"] = df["risk_labels"].apply(parse_json_list)
df["risk_explanations_list"] = df["risk_explanations"].apply(parse_json_list)

flat_tags = sorted({tag for tags in df["tags_list"] for tag in tags})
flat_risk = sorted({label for labels in df["risk_labels_list"] for label in labels})

risk_label_display = {
    "toxicity": "Toxicity",
    "pii": "PII",
    "self_harm": "Self-Harm",
    "jailbreak": "Jailbreak",
    "bias": "Bias",
    "refusal": "Refusal",
}

df["risk_reason"] = df["risk_labels_list"].apply(
    lambda labels: ", ".join(risk_label_display.get(label, label) for label in labels)
    or "None"
)

df["status"] = df["flagged"].map({True: "Flagged", False: "Safe"})

with st.sidebar:
    st.header("Filters")
    search_query = st.text_input("Search", "", placeholder="Search input/output text")
    statuses = st.multiselect(
        "Status", ["Flagged", "Safe"], default=["Flagged", "Safe"]
    )
    min_toxicity = st.slider("Minimum toxicity", 0.0, 1.0, 0.0, 0.05)
    project_filter = st.multiselect(
        "Project", sorted([p for p in df["project_name"].dropna().unique() if p])
    )
    model_filter = st.multiselect(
        "Model", sorted([m for m in df["model_name"].dropna().unique() if m])
    )
    user_filter = st.multiselect(
        "User ID", sorted([u for u in df["user_id"].dropna().unique() if u])
    )
    tag_filter = st.multiselect("Tags", flat_tags)
    risk_filter = st.multiselect("Risk labels", flat_risk)
    pii_only = st.checkbox("PII only", value=False)
    refusal_only = st.checkbox("Refusal only", value=False)
    if df["timestamp"].notna().any():
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date))
    else:
        date_range = None

filtered = df[df["status"].isin(statuses)].copy()

if search_query:
    mask = filtered["input_text"].str.contains(
        search_query, case=False, na=False
    ) | filtered["output_text"].str.contains(search_query, case=False, na=False)
    filtered = filtered[mask]

if min_toxicity > 0:
    filtered = filtered[filtered["toxicity_score"] >= min_toxicity]

if project_filter:
    filtered = filtered[filtered["project_name"].isin(project_filter)]

if model_filter:
    filtered = filtered[filtered["model_name"].isin(model_filter)]

if user_filter:
    filtered = filtered[filtered["user_id"].isin(user_filter)]

if tag_filter:
    filtered = filtered[
        filtered["tags_list"].apply(lambda tags: any(tag in tags for tag in tag_filter))
    ]

if risk_filter:
    filtered = filtered[
        filtered["risk_labels_list"].apply(
            lambda labels: any(label in labels for label in risk_filter)
        )
    ]

if pii_only:
    filtered = filtered[filtered["has_pii"]]

if refusal_only:
    filtered = filtered[filtered["is_refusal"]]

if df["timestamp"].notna().any() and date_range:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["timestamp"].dt.date.between(start_date, end_date, inclusive="both")
    ]

st.subheader("Overview")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
metric_col1.metric("Total Audited", len(df))
metric_col2.metric("Flagged Risks", int(df["flagged"].sum()), delta_color="inverse")
metric_col3.metric("Refusal Rate", f"{df['is_refusal'].mean() * 100:.1f}%")
metric_col4.metric("Self-Harm Rate", f"{df['self_harm'].mean() * 100:.1f}%")
metric_col5.metric("Avg Toxicity", f"{df['toxicity_score'].mean():.3f}")

st.caption(f"Showing {len(filtered)} of {len(df)} records after filters.")

st.subheader("Risk Analysis")
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.write("Toxicity Distribution")
    bins = pd.cut(
        filtered["toxicity_score"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    )
    toxicity_counts = bins.value_counts().sort_index()
    toxicity_df = toxicity_counts.reset_index()
    toxicity_df.columns = ["toxicity_bin", "count"]
    toxicity_df["toxicity_bin"] = toxicity_df["toxicity_bin"].astype(str)
    st.bar_chart(toxicity_df, x="toxicity_bin", y="count")

with chart_col2:
    st.write("Flagged vs Safe")
    status_counts = filtered["status"].value_counts()
    st.bar_chart(status_counts)

with chart_col3:
    st.write("Risk Labels")
    label_counts = (
        filtered["risk_labels_list"]
        .explode()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    if not label_counts.empty:
        st.bar_chart(label_counts)
    else:
        st.caption("No risk labels in current filter.")

signal_breakdown = (
    filtered["risk_labels_list"]
    .explode()
    .replace("", pd.NA)
    .dropna()
    .value_counts()
    .rename_axis("signal")
    .reset_index(name="count")
)
if not signal_breakdown.empty:
    signal_breakdown["percent"] = (
        signal_breakdown["count"] / signal_breakdown["count"].sum() * 100
    ).round(1)
    st.subheader("Signal Breakdown")
    st.dataframe(signal_breakdown, use_container_width=True)

if filtered["timestamp"].notna().any():
    st.subheader("Audit Volume Over Time")
    time_series = (
        filtered.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("D")
        .size()
    )
    st.line_chart(time_series)

st.subheader("Audit Logs")

if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

csv_data = filtered.to_csv(index=False)
st.download_button("Download CSV", csv_data, file_name="audit_logs_filtered.csv")

st.dataframe(
    filtered[
        [
            "timestamp",
            "project_name",
            "model_name",
            "user_id",
            "input_text",
            "output_text",
            "toxicity_score",
            "has_pii",
            "is_refusal",
            "self_harm",
            "jailbreak",
            "bias",
            "sentiment_score",
            "flagged",
            "risk_reason",
        ]
    ],
    use_container_width=True,
)

st.subheader("Record Details")
if not filtered.empty:
    record_id = st.selectbox("Select record ID", filtered["id"].tolist())
    record = filtered[filtered["id"] == record_id].iloc[0]
    st.write(f"**Project:** {record['project_name'] or 'N/A'}")
    st.write(f"**Model:** {record['model_name'] or 'N/A'}")
    st.write(f"**User ID:** {record['user_id'] or 'N/A'}")
    st.write(f"**Request ID:** {record['request_id'] or 'N/A'}")
    st.write(f"**Tags:** {', '.join(record['tags_list']) or 'N/A'}")
    st.write(f"**Risk Labels:** {record['risk_reason']}")
    if record["risk_explanations_list"]:
        st.write("**Why flagged**")
        for explanation in record["risk_explanations_list"]:
            st.write(f"- {explanation}")
    if record.get("redaction_applied"):
        st.write(
            f"**PII Redaction:** applied ({int(record.get('redaction_count', 0))} match(es))"
        )
    st.write("**Input**")
    st.code(record["input_text"], language="text")
    st.write("**Output**")
    st.code(record["output_text"], language="text")
else:
    st.caption("No records to display.")

st.subheader("Import Data")

with st.expander("Upload CSV/JSONL for auditing"):
    uploaded = st.file_uploader("Select a file", type=["csv", "jsonl"])
    col_a, col_b = st.columns(2)
    with col_a:
        upload_project = st.text_input("Default project", value="")
        upload_model = st.text_input("Default model", value="")
        upload_user = st.text_input("Default user id", value="")
    with col_b:
        upload_request = st.text_input("Default request id", value="")
        upload_tags = st.text_input("Default tags (comma separated)", value="")
        disable_tox = st.checkbox("Disable toxicity model for upload", value=True)

    if uploaded is not None:
        if st.button("Ingest file"):
            try:
                if uploaded.name.endswith(".csv"):
                    data_df = pd.read_csv(uploaded)
                    records = [
                        ConversationRecord(
                            input_text=str(row.get("input_text", "")),
                            output_text=str(row.get("output_text", "")),
                            project_name=row.get("project_name") or upload_project,
                            model_name=row.get("model_name") or upload_model,
                            user_id=row.get("user_id") or upload_user,
                            request_id=row.get("request_id") or upload_request,
                            tags=normalize_tags(row.get("tags"))
                            or normalize_tags(upload_tags),
                            timestamp=row.get("timestamp"),
                        )
                        for _, row in data_df.iterrows()
                    ]
                else:
                    records = []
                    for line in uploaded.getvalue().decode("utf-8").splitlines():
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        record = ConversationRecord(
                            input_text=str(payload.get("input_text", "")),
                            output_text=str(payload.get("output_text", "")),
                            project_name=payload.get("project_name") or upload_project,
                            model_name=payload.get("model_name") or upload_model,
                            user_id=payload.get("user_id") or upload_user,
                            request_id=payload.get("request_id") or upload_request,
                            tags=normalize_tags(payload.get("tags"))
                            or normalize_tags(upload_tags),
                            timestamp=payload.get("timestamp"),
                        )
                        records.append(record)

                detector = None
                if disable_tox:
                    detector = SignalDetector(enable_toxicity=False)
                with AuditEngine(db_path=db_path, detector=detector) as engine:
                    engine.process_batch(records)

                st.success(f"Ingested {len(records)} records into {db_path}.")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to ingest data: {exc}")
