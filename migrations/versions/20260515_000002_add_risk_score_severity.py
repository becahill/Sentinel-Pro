"""add risk score and severity fields

Revision ID: 20260515_000002
Revises: 20260202_000001
Create Date: 2026-05-15 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "20260515_000002"
down_revision = "20260202_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("audit_logs") as batch_op:
        batch_op.add_column(sa.Column("risk_score", sa.Float()))
        batch_op.add_column(sa.Column("severity", sa.Text()))
        batch_op.add_column(sa.Column("detector_results", sa.Text()))
    op.create_index("ix_audit_logs_severity", "audit_logs", ["severity"])


def downgrade() -> None:
    op.drop_index("ix_audit_logs_severity", table_name="audit_logs")
    with op.batch_alter_table("audit_logs") as batch_op:
        batch_op.drop_column("detector_results")
        batch_op.drop_column("severity")
        batch_op.drop_column("risk_score")
