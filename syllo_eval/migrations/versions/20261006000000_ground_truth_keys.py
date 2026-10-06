"""Metrics declare several ground-truth keys, and computations link to every ground truth they used."""

from alembic import op

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
  op.execute("""
    CREATE TABLE metric_computation_ground_truth (
        metric_computation_id UUID NOT NULL,
        ground_truth_id UUID NOT NULL,
        CONSTRAINT pk_metric_computation_ground_truth PRIMARY KEY (metric_computation_id, ground_truth_id),
        CONSTRAINT fk_mcgt_metric_computation FOREIGN KEY (metric_computation_id)
          REFERENCES span_metric_computation(id) ON DELETE CASCADE,
        CONSTRAINT fk_mcgt_ground_truth FOREIGN KEY (ground_truth_id) REFERENCES ground_truth(id) ON DELETE RESTRICT
    );
    CREATE INDEX idx_mcgt_ground_truth_id ON metric_computation_ground_truth(ground_truth_id);
    COMMENT ON TABLE metric_computation_ground_truth IS 'The ground truths a metric computation used.';
    INSERT INTO metric_computation_ground_truth (metric_computation_id, ground_truth_id)
      SELECT id, ground_truth_id FROM span_metric_computation WHERE ground_truth_id IS NOT NULL;
    ALTER TABLE span_metric_computation DROP COLUMN ground_truth_id;

    ALTER TABLE metric ADD COLUMN ground_truth_keys TEXT[] NOT NULL DEFAULT '{}';
    COMMENT ON COLUMN metric.ground_truth_keys IS 'The ground-truth keys this metric reads.';
    UPDATE metric SET ground_truth_keys = ARRAY[ground_truth_key] WHERE ground_truth_key IS NOT NULL;
    ALTER TABLE metric DROP COLUMN ground_truth_key;
    ALTER TABLE metric DROP COLUMN requires_ground_truth;
  """)


def downgrade() -> None:
  # A computation linked to several ground truths keeps only one of them.
  op.execute("""
    ALTER TABLE metric ADD COLUMN requires_ground_truth BOOLEAN NOT NULL DEFAULT TRUE;
    ALTER TABLE metric ADD COLUMN ground_truth_key TEXT;
    UPDATE metric
      SET ground_truth_key = ground_truth_keys[1], requires_ground_truth = cardinality(ground_truth_keys) > 0;
    ALTER TABLE metric DROP COLUMN ground_truth_keys;

    ALTER TABLE span_metric_computation ADD COLUMN ground_truth_id UUID;
    ALTER TABLE span_metric_computation ADD CONSTRAINT fk_smc_ground_truth
      FOREIGN KEY (ground_truth_id) REFERENCES ground_truth(id) ON DELETE RESTRICT;
    CREATE INDEX idx_smc_ground_truth_id ON span_metric_computation(ground_truth_id);
    UPDATE span_metric_computation smc SET ground_truth_id = (
      SELECT MIN(mcgt.ground_truth_id::text)::uuid FROM metric_computation_ground_truth mcgt
      WHERE mcgt.metric_computation_id = smc.id
    );
    DROP TABLE metric_computation_ground_truth;
  """)
