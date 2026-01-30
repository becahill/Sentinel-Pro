import json
import sqlite3

import pandas as pd

from auditor import AuditEngine
from signals import SignalDetector


def test_audit_engine_writes_row(tmp_path):
    db_path = tmp_path / "audit_logs.db"
    detector = SignalDetector(enable_toxicity=False)

    with AuditEngine(db_path=str(db_path), detector=detector) as engine:
        flagged = engine.process_transaction("Hello", "Contact admin@corp.com")

    assert flagged is True

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql("SELECT * FROM audit_logs", conn)
    finally:
        conn.close()

    assert len(df) == 1
    assert bool(df.loc[0, "has_pii"]) is True
    risk_labels = json.loads(df.loc[0, "risk_labels"])
    assert "pii" in risk_labels
    assert "[REDACTED_EMAIL]" in df.loc[0, "output_text"]
    explanations = json.loads(df.loc[0, "risk_explanations"])
    assert any("PII" in explanation for explanation in explanations)
