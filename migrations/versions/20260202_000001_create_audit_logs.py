"""create audit logs

Revision ID: 20260202_000001
Revises:
Create Date: 2026-02-02 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "20260202_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.Text(), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False),
        sa.Column("toxicity_score", sa.Float()),
        sa.Column("has_pii", sa.Boolean()),
        sa.Column("is_refusal", sa.Boolean()),
        sa.Column("self_harm", sa.Boolean()),
        sa.Column("jailbreak", sa.Boolean()),
        sa.Column("bias", sa.Boolean()),
        sa.Column("sentiment_score", sa.Float()),
        sa.Column("risk_labels", sa.Text()),
        sa.Column("risk_explanations", sa.Text()),
        sa.Column("pii_types", sa.Text()),
        sa.Column("flagged", sa.Boolean()),
        sa.Column("redaction_applied", sa.Boolean()),
        sa.Column("redaction_count", sa.Integer()),
        sa.Column("project_name", sa.Text()),
        sa.Column("model_name", sa.Text()),
        sa.Column("user_id", sa.Text()),
        sa.Column("request_id", sa.Text()),
        sa.Column("tags", sa.Text()),
    )
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("ix_audit_logs_flagged", "audit_logs", ["flagged"])


def downgrade() -> None:
    op.drop_index("ix_audit_logs_flagged", table_name="audit_logs")
    op.drop_index("ix_audit_logs_timestamp", table_name="audit_logs")
    op.drop_table("audit_logs")
