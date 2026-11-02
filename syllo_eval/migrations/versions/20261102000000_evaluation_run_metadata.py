"""Caller-supplied run metadata, separate from the resolved run config."""

from alembic import op

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
  op.execute("""
    ALTER TABLE evaluation_run ADD COLUMN metadata JSONB CHECK (jsonb_typeof(metadata) = 'object');
  """)


def downgrade() -> None:
  op.execute("""
    ALTER TABLE evaluation_run DROP COLUMN metadata;
  """)
