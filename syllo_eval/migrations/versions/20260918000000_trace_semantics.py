"""Typed canonical trace observations; existing span payloads are preserved as JSON."""

from alembic import op

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
  op.execute("""
    ALTER TABLE trace ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE trace ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown';
    ALTER TABLE trace ADD COLUMN adapter TEXT;
    ALTER TABLE span ADD COLUMN status TEXT NOT NULL DEFAULT 'unset'
      CHECK (status IN ('success', 'error', 'unset'));
    ALTER TABLE span ADD COLUMN semantics JSONB NOT NULL DEFAULT '{}';
    ALTER TABLE span ALTER COLUMN input_data TYPE JSONB USING to_jsonb(input_data);
    ALTER TABLE span ALTER COLUMN output_data TYPE JSONB USING to_jsonb(output_data);
  """)


def downgrade() -> None:
  op.execute("""
    ALTER TABLE span ALTER COLUMN input_data TYPE TEXT USING
      CASE WHEN jsonb_typeof(input_data) = 'string' THEN input_data #>> '{}' ELSE input_data::text END;
    ALTER TABLE span ALTER COLUMN output_data TYPE TEXT USING
      CASE WHEN jsonb_typeof(output_data) = 'string' THEN output_data #>> '{}' ELSE output_data::text END;
    ALTER TABLE span DROP COLUMN semantics;
    ALTER TABLE span DROP COLUMN status;
    ALTER TABLE trace DROP COLUMN adapter;
    ALTER TABLE trace DROP COLUMN source;
    ALTER TABLE trace DROP COLUMN schema_version;
  """)
