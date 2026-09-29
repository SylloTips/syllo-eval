# ruff: noqa: E501
"""Baseline schema.

Revision ID: 0001
Revises:
Create Date: 2026-07-22 00:00:00

"""

from alembic import op

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
  op.execute("""CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE evaluation_run_status AS ENUM ('RUNNING', 'COMPLETED', 'PARTIALLY_COMPLETED', 'FAILED');
CREATE TYPE evaluation_sample_status AS ENUM ('RUNNING', 'COMPLETED', 'FAILED');
CREATE TYPE metric_computation_status AS ENUM ('COMPLETED', 'FAILED', 'SKIPPED');
CREATE TYPE metric_targeting_mode AS ENUM ('SINGLE', 'GROUP');

CREATE TABLE agent (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    version_tag TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT uk_agent_name_version UNIQUE (name, version_tag)
);

CREATE TABLE dataset (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE sample (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL,
    input_prompt TEXT NOT NULL,
    ground_truth_output TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_sample_dataset FOREIGN KEY (dataset_id) REFERENCES dataset(id) ON DELETE CASCADE
);

CREATE TABLE metric (
    name TEXT PRIMARY KEY,
    description TEXT,
    requires_ground_truth BOOLEAN NOT NULL DEFAULT TRUE,
    ground_truth_key TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE span_type (
    name TEXT PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE metric_target_span_type (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric TEXT NOT NULL,
    span_type TEXT NOT NULL,
    targeting_mode metric_targeting_mode NOT NULL DEFAULT 'SINGLE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_mtst_metric FOREIGN KEY (metric) REFERENCES metric(name) ON DELETE CASCADE,
    CONSTRAINT fk_mtst_span_type FOREIGN KEY (span_type) REFERENCES span_type(name) ON DELETE CASCADE,
    CONSTRAINT uk_metric_span_type UNIQUE (metric, span_type)
);

CREATE TABLE ground_truth (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id UUID NOT NULL,
    key TEXT NOT NULL,
    ground_truth_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_ground_truth_sample FOREIGN KEY (sample_id) REFERENCES sample(id) ON DELETE CASCADE,
    CONSTRAINT uk_ground_truth_sample_key UNIQUE (sample_id, key)
);

CREATE TABLE evaluation_run (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    dataset_id UUID NOT NULL,
    status evaluation_run_status NOT NULL DEFAULT 'RUNNING',
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    source_run_id UUID,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_evaluation_run_agent FOREIGN KEY (agent_id) REFERENCES agent(id) ON DELETE CASCADE,
    CONSTRAINT fk_evaluation_run_dataset FOREIGN KEY (dataset_id) REFERENCES dataset(id) ON DELETE CASCADE,
    CONSTRAINT fk_evaluation_run_source_run FOREIGN KEY (source_run_id) REFERENCES evaluation_run(id) ON DELETE SET NULL
);

CREATE TABLE evaluation_run_metric (
    evaluation_run_id UUID NOT NULL,
    metric TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT pk_evaluation_run_metric PRIMARY KEY (evaluation_run_id, metric),
    CONSTRAINT fk_erm_evaluation_run FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_run(id) ON DELETE CASCADE,
    CONSTRAINT fk_erm_metric FOREIGN KEY (metric) REFERENCES metric(name) ON DELETE CASCADE
);

CREATE TABLE evaluation_run_plan_sample (
    evaluation_run_id UUID NOT NULL,
    sample_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT pk_evaluation_run_plan_sample PRIMARY KEY (evaluation_run_id, sample_id),
    CONSTRAINT fk_erps_evaluation_run FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_run(id) ON DELETE CASCADE,
    CONSTRAINT fk_erps_sample FOREIGN KEY (sample_id) REFERENCES sample(id) ON DELETE CASCADE
);

CREATE TABLE trace (
    external_id TEXT PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE span (
    external_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    span_type TEXT NOT NULL,
    name TEXT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    input_data TEXT NOT NULL,
    output_data TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_span_trace FOREIGN KEY (trace_id) REFERENCES trace(external_id) ON DELETE CASCADE,
    CONSTRAINT fk_span_parent FOREIGN KEY (parent_span_id) REFERENCES span(external_id) ON DELETE CASCADE,
    CONSTRAINT fk_span_span_type FOREIGN KEY (span_type) REFERENCES span_type(name) ON DELETE RESTRICT
);

CREATE TABLE evaluation_run_sample (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_run_id UUID NOT NULL,
    sample_id UUID NOT NULL,
    trace_id TEXT,
    status evaluation_sample_status NOT NULL DEFAULT 'RUNNING',
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_ers_evaluation_run FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_run(id) ON DELETE CASCADE,
    CONSTRAINT fk_ers_sample FOREIGN KEY (sample_id) REFERENCES sample(id) ON DELETE CASCADE,
    CONSTRAINT fk_ers_trace FOREIGN KEY (trace_id) REFERENCES trace(external_id) ON DELETE CASCADE,
    CONSTRAINT uk_evaluation_run_sample UNIQUE (evaluation_run_id, sample_id)
);

CREATE TABLE span_metric_computation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_run_sample_id UUID NOT NULL,
    metric TEXT NOT NULL,
    ground_truth_id UUID,
    targeting_mode metric_targeting_mode NOT NULL DEFAULT 'SINGLE',
    target_span_type TEXT NOT NULL,
    score DOUBLE PRECISION,
    status metric_computation_status NOT NULL DEFAULT 'COMPLETED',
    reasoning TEXT,
    metadata JSONB,
    error_message TEXT,
    raw_output JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_smc_evaluation_sample_result FOREIGN KEY (evaluation_run_sample_id) REFERENCES evaluation_run_sample(id) ON DELETE CASCADE,
    CONSTRAINT fk_smc_metric FOREIGN KEY (metric) REFERENCES metric(name) ON DELETE CASCADE,
    CONSTRAINT fk_smc_ground_truth FOREIGN KEY (ground_truth_id) REFERENCES ground_truth(id) ON DELETE RESTRICT,
    CONSTRAINT fk_smc_target_span_type FOREIGN KEY (target_span_type) REFERENCES span_type(name) ON DELETE RESTRICT
);

CREATE TABLE single_span_metric_computation (
    metric_computation_id UUID PRIMARY KEY,
    span_id TEXT NOT NULL,
    CONSTRAINT fk_ssmc_metric_computation FOREIGN KEY (metric_computation_id) REFERENCES span_metric_computation(id) ON DELETE CASCADE,
    CONSTRAINT fk_ssmc_span FOREIGN KEY (span_id) REFERENCES span(external_id) ON DELETE RESTRICT
);

CREATE TABLE group_span_metric_computation_span (
    metric_computation_id UUID NOT NULL,
    span_id TEXT NOT NULL,
    CONSTRAINT pk_group_span_metric_computation_span PRIMARY KEY (metric_computation_id, span_id),
    CONSTRAINT fk_gsmcs_metric_computation FOREIGN KEY (metric_computation_id) REFERENCES span_metric_computation(id) ON DELETE CASCADE,
    CONSTRAINT fk_gsmcs_span FOREIGN KEY (span_id) REFERENCES span(external_id) ON DELETE RESTRICT
);

COMMENT ON COLUMN agent.id IS 'The unique ID of the Agent.';
COMMENT ON COLUMN agent.name IS 'The name of the Agent.';
COMMENT ON COLUMN agent.version_tag IS 'The version tag of the Agent.';
COMMENT ON COLUMN agent.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN agent.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN dataset.id IS 'The unique ID of the dataset.';
COMMENT ON COLUMN dataset.name IS 'The name of the dataset.';
COMMENT ON COLUMN dataset.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN dataset.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN sample.id IS 'The unique identifier of the sample.';
COMMENT ON COLUMN sample.dataset_id IS 'The id of the dataset this sample belongs to.';
COMMENT ON COLUMN sample.input_prompt IS 'The input prompt of the sample.';
COMMENT ON COLUMN sample.ground_truth_output IS 'The expected output of the sample.';
COMMENT ON COLUMN sample.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN sample.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN metric.name IS 'The name of the metric.';
COMMENT ON COLUMN metric.description IS 'The description of the metric (nullable).';
COMMENT ON COLUMN metric.requires_ground_truth IS 'Whether the metric requires ground truth for computation.';
COMMENT ON COLUMN metric.ground_truth_key IS 'The ground-truth key consumed by this metric (nullable).';
COMMENT ON COLUMN metric.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN metric.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN span_type.name IS 'The name of the span type.';
COMMENT ON COLUMN span_type.description IS 'The description of the span type (nullable).';
COMMENT ON COLUMN span_type.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN span_type.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN metric_target_span_type.id IS 'The unique identifier of the target.';
COMMENT ON COLUMN metric_target_span_type.metric IS 'The name of the metric that this span type is a target of.';
COMMENT ON COLUMN metric_target_span_type.span_type IS 'The span type.';
COMMENT ON COLUMN metric_target_span_type.targeting_mode IS 'Whether the metric targets one span or a span group for this span type.';
COMMENT ON COLUMN metric_target_span_type.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN metric_target_span_type.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN ground_truth.id IS 'The unique identifier of the ground truth.';
COMMENT ON COLUMN ground_truth.sample_id IS 'The id of the sample this ground truth belongs to.';
COMMENT ON COLUMN ground_truth.key IS 'The kind of ground truth (expected_output, expected_plan, relevant_document_ids, relevant_snippet_ids).';
COMMENT ON COLUMN ground_truth.ground_truth_value IS 'The value of the ground truth, a JSON value.';
COMMENT ON COLUMN ground_truth.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN ground_truth.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN evaluation_run.id IS 'The unique identifier of the evaluation run.';
COMMENT ON COLUMN evaluation_run.agent_id IS 'The id of the agent that is being evaluated.';
COMMENT ON COLUMN evaluation_run.dataset_id IS 'The id of the dataset used in this evaluation run.';
COMMENT ON COLUMN evaluation_run.status IS 'The status of the evaluation run.';
COMMENT ON COLUMN evaluation_run.start_time IS 'The start time of the evaluation run.';
COMMENT ON COLUMN evaluation_run.end_time IS 'The end time of the evaluation run (nullable).';
COMMENT ON COLUMN evaluation_run.source_run_id IS 'The source evaluation run this run repeats (nullable).';
COMMENT ON COLUMN evaluation_run.config IS 'Scalar execution configuration snapshot for the run (nullable).';
COMMENT ON COLUMN evaluation_run.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN evaluation_run.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN evaluation_run_metric.evaluation_run_id IS 'The evaluation run whose metric plan is snapshotted.';
COMMENT ON COLUMN evaluation_run_metric.metric IS 'The metric selected for this evaluation run.';
COMMENT ON COLUMN evaluation_run_metric.created_at IS 'Timestamp when the record was created.';

COMMENT ON COLUMN evaluation_run_plan_sample.evaluation_run_id IS 'The evaluation run whose sample plan is snapshotted.';
COMMENT ON COLUMN evaluation_run_plan_sample.sample_id IS 'The sample selected for this evaluation run.';
COMMENT ON COLUMN evaluation_run_plan_sample.created_at IS 'Timestamp when the record was created.';

COMMENT ON COLUMN trace.external_id IS 'The unique identifier of the trace (external trace-source ID).';
COMMENT ON COLUMN trace.start_time IS 'The start time of the trace.';
COMMENT ON COLUMN trace.end_time IS 'The end time of the trace.';
COMMENT ON COLUMN trace.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN trace.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN span.external_id IS 'The unique identifier of the span (external trace-source ID).';
COMMENT ON COLUMN span.trace_id IS 'The id of the trace this span belongs to.';
COMMENT ON COLUMN span.parent_span_id IS 'The id of the parent span (nullable for root spans).';
COMMENT ON COLUMN span.span_type IS 'The id of the span type.';
COMMENT ON COLUMN span.name IS 'The name of the span.';
COMMENT ON COLUMN span.start_time IS 'The start time of the span.';
COMMENT ON COLUMN span.end_time IS 'The end time of the span.';
COMMENT ON COLUMN span.input_data IS 'The input of the span.';
COMMENT ON COLUMN span.output_data IS 'The output of the span.';
COMMENT ON COLUMN span.metadata IS 'Additional metadata for the span, e.g., LLM input/output messages and tool calls (nullable).';
COMMENT ON COLUMN span.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN span.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN evaluation_run_sample.id IS 'The unique identifier of the evaluation run sample.';
COMMENT ON COLUMN evaluation_run_sample.evaluation_run_id IS 'The id of the evaluation run this sample belongs to.';
COMMENT ON COLUMN evaluation_run_sample.sample_id IS 'The id of the sample this run belongs to.';
COMMENT ON COLUMN evaluation_run_sample.trace_id IS 'The trace id for this sample''s execution (external trace-source ID), nullable when execution fails before trace creation.';
COMMENT ON COLUMN evaluation_run_sample.status IS 'The execution status of this sample within the evaluation run.';
COMMENT ON COLUMN evaluation_run_sample.started_at IS 'Timestamp when sample execution started (nullable).';
COMMENT ON COLUMN evaluation_run_sample.ended_at IS 'Timestamp when sample execution ended (nullable).';
COMMENT ON COLUMN evaluation_run_sample.error_message IS 'Error message recorded when sample execution fails or is skipped (nullable).';
COMMENT ON COLUMN evaluation_run_sample.metadata IS 'Additional structured metadata for this sample execution (nullable).';
COMMENT ON COLUMN evaluation_run_sample.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN evaluation_run_sample.updated_at IS 'Timestamp when the record was last updated.';

COMMENT ON COLUMN span_metric_computation.id IS 'The unique identifier of the metric computation.';
COMMENT ON COLUMN span_metric_computation.evaluation_run_sample_id IS 'The id of the evaluation run sample that this computation belongs to.';
COMMENT ON COLUMN span_metric_computation.metric IS 'The name of the metric that is being computed.';
COMMENT ON COLUMN span_metric_computation.ground_truth_id IS 'The ground truth row used by this computation (nullable).';
COMMENT ON COLUMN span_metric_computation.targeting_mode IS 'Whether the metric targeted one span or a span group.';
COMMENT ON COLUMN span_metric_computation.target_span_type IS 'The configured span type targeted by this computation, even when no span was present.';
COMMENT ON COLUMN span_metric_computation.score IS 'The score of a completed computation (nullable for failed or skipped computations).';
COMMENT ON COLUMN span_metric_computation.status IS 'The execution status of this metric computation.';
COMMENT ON COLUMN span_metric_computation.reasoning IS 'Explanation for the score (nullable).';
COMMENT ON COLUMN span_metric_computation.metadata IS 'Additional metadata for the computation, e.g., error details (nullable).';
COMMENT ON COLUMN span_metric_computation.error_message IS 'Error message recorded when metric computation fails or is skipped (nullable).';
COMMENT ON COLUMN span_metric_computation.raw_output IS 'Raw structured output returned by a metric or judge (nullable).';
COMMENT ON COLUMN span_metric_computation.created_at IS 'Timestamp when the record was created.';
COMMENT ON COLUMN span_metric_computation.updated_at IS 'Timestamp when the record was last updated.';
COMMENT ON COLUMN single_span_metric_computation.metric_computation_id IS 'The metric computation header for a single-span computation.';
COMMENT ON COLUMN single_span_metric_computation.span_id IS 'The single span evaluated by the computation.';
COMMENT ON COLUMN group_span_metric_computation_span.metric_computation_id IS 'The metric computation header for a group computation.';
COMMENT ON COLUMN group_span_metric_computation_span.span_id IS 'One span included in the group computation.';

CREATE INDEX idx_agent_name ON agent(name);
CREATE INDEX idx_agent_version_tag ON agent(version_tag);

CREATE INDEX idx_dataset_name ON dataset(name);

CREATE INDEX idx_sample_dataset_id ON sample(dataset_id);

CREATE INDEX idx_mtst_metric ON metric_target_span_type(metric);
CREATE INDEX idx_mtst_span_type ON metric_target_span_type(span_type);
CREATE INDEX idx_mtst_targeting_mode ON metric_target_span_type(targeting_mode);

CREATE INDEX idx_ground_truth_sample_id ON ground_truth(sample_id);
CREATE INDEX idx_ground_truth_key ON ground_truth(key);

CREATE INDEX idx_evaluation_run_agent_id ON evaluation_run(agent_id);
CREATE INDEX idx_evaluation_run_dataset_id ON evaluation_run(dataset_id);
CREATE INDEX idx_evaluation_run_status ON evaluation_run(status);
CREATE INDEX idx_evaluation_run_start_time ON evaluation_run(start_time);
CREATE INDEX idx_evaluation_run_source_run_id ON evaluation_run(source_run_id);

CREATE INDEX idx_erm_metric ON evaluation_run_metric(metric);

CREATE INDEX idx_erps_sample_id ON evaluation_run_plan_sample(sample_id);

CREATE INDEX idx_trace_start_time ON trace(start_time);

CREATE INDEX idx_span_trace_id ON span(trace_id);
CREATE INDEX idx_span_parent_span_id ON span(parent_span_id);
CREATE INDEX idx_span_span_type ON span(span_type);
CREATE INDEX idx_span_start_time ON span(start_time);

CREATE INDEX idx_ers_evaluation_run_id ON evaluation_run_sample(evaluation_run_id);
CREATE INDEX idx_ers_sample_id ON evaluation_run_sample(sample_id);
CREATE INDEX idx_ers_trace_id ON evaluation_run_sample(trace_id);
CREATE INDEX idx_ers_status ON evaluation_run_sample(status);

CREATE INDEX idx_smc_evaluation_run_sample_id ON span_metric_computation(evaluation_run_sample_id);
CREATE INDEX idx_smc_metric ON span_metric_computation(metric);
CREATE INDEX idx_smc_ground_truth_id ON span_metric_computation(ground_truth_id);
CREATE INDEX idx_smc_targeting_mode ON span_metric_computation(targeting_mode);
CREATE INDEX idx_smc_target_span_type ON span_metric_computation(target_span_type);
CREATE INDEX idx_smc_score ON span_metric_computation(score);
CREATE INDEX idx_smc_status ON span_metric_computation(status);
CREATE INDEX idx_ssmc_span_id ON single_span_metric_computation(span_id);
CREATE INDEX idx_gsmcs_span_id ON group_span_metric_computation_span(span_id);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agent_updated_at BEFORE UPDATE ON agent
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dataset_updated_at BEFORE UPDATE ON dataset
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sample_updated_at BEFORE UPDATE ON sample
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_metric_updated_at BEFORE UPDATE ON metric
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_span_type_updated_at BEFORE UPDATE ON span_type
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_metric_target_span_type_updated_at BEFORE UPDATE ON metric_target_span_type
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ground_truth_updated_at BEFORE UPDATE ON ground_truth
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_evaluation_run_updated_at BEFORE UPDATE ON evaluation_run
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trace_updated_at BEFORE UPDATE ON trace
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_span_updated_at BEFORE UPDATE ON span
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_evaluation_run_sample_updated_at BEFORE UPDATE ON evaluation_run_sample
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_span_metric_computation_updated_at BEFORE UPDATE ON span_metric_computation
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
""")


def downgrade() -> None:
  raise NotImplementedError('The baseline migration cannot be downgraded safely.')
