"""This module provides data model classes that are useful for all evaluation runs."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from syllo_eval.trace_semantics import SpanSemantics


class EvaluationStatus(str, Enum):
  RUNNING = 'RUNNING'
  COMPLETED = 'COMPLETED'
  PARTIALLY_COMPLETED = 'PARTIALLY_COMPLETED'
  FAILED = 'FAILED'


class EvaluationSampleStatus(str, Enum):
  RUNNING = 'RUNNING'
  COMPLETED = 'COMPLETED'
  FAILED = 'FAILED'


class MetricComputationStatus(str, Enum):
  COMPLETED = 'COMPLETED'
  FAILED = 'FAILED'
  SKIPPED = 'SKIPPED'


class MetricTargetingMode(str, Enum):
  SINGLE = 'SINGLE'
  GROUP = 'GROUP'


class Agent(BaseModel):
  """The Agent that is being evaluated.

  Attributes:
      - id: The unique ID of the Agent.
      - name: The name of the Agent.
      - version_tag: The version tag of the Agent.
  """

  id: UUID
  name: str
  version_tag: str


class Dataset(BaseModel):
  """A dataset is a collection of test cases (samples).

  Attributes:
      - id: The unique ID of the dataset.
      - name: The name of the dataset.
  """

  id: UUID
  name: str


class DatasetSummary(BaseModel):
  """A dataset listing projection with its sample count.

  Attributes:
      - id: The unique ID of the dataset.
      - name: The name of the dataset.
      - created_at: Timestamp when the dataset was created.
      - sample_count: Number of samples in the dataset.
  """

  id: UUID
  name: str
  created_at: datetime
  sample_count: int


class Sample(BaseModel):
  """Represents a single test sample (one test case).

  Attributes:
      - id: The unique identifier of the sample.
      - dataset_id: The id of the dataset this sample belongs to.
      - input_prompt: The input prompt of the sample.
      - ground_truth_output: The expected output of the sample.
  """

  id: UUID
  dataset_id: UUID
  input_prompt: str
  ground_truth_output: str | None = None


class GroundTruthKey(str, Enum):
  """Kinds of ground truth a sample can carry; metrics declare which kind they consume."""

  EXPECTED_OUTPUT = 'expected_output'
  EXPECTED_PLAN = 'expected_plan'
  EXPECTED_CLAIMS = 'expected_claims'
  RELEVANT_DOCUMENT_IDS = 'relevant_document_ids'
  RELEVANT_SNIPPET_IDS = 'relevant_snippet_ids'


class ExpectedClaim(BaseModel):
  """One atomic claim of an expected answer, stored as ground truth: ``{"claims": [{"id", "text"}, ...]}``."""

  model_config = ConfigDict(extra='forbid')

  id: str = Field(min_length=1)
  text: str = Field(min_length=1)


class GroundTruth(BaseModel):
  """Represents the ground truth output of a sample.

  Attributes:
      - id: The unique identifier of the ground truth.
      - sample_id: The id of the sample this ground truth belongs to.
      - key: The kind of ground truth (see GroundTruthKey); one row per (sample, key).
      - ground_truth_value: The value of the ground truth, a JSON value.
  """

  id: UUID
  sample_id: UUID
  key: str
  ground_truth_value: dict[str, Any]


class EvaluationRun(BaseModel):
  """Represents a single evaluation run (e.g., evaluating agent version v1 on the 'dataset_1' dataset).

  Attributes:
      - id: The unique identifier of the evaluation run.
      - agent_id: The id of the agent that is being evaluated.
      - dataset_id: The id of the dataset used in this evaluation run.
      - status: The status of the evaluation run.
      - start_time: The start time of the evaluation run.
      - end_time: The end time of the evaluation run (nullable).
  """

  id: UUID
  agent_id: UUID
  dataset_id: UUID
  status: EvaluationStatus  # {RUNNING, COMPLETED, PARTIALLY_COMPLETED, FAILED}
  start_time: datetime
  end_time: datetime | None = None
  source_run_id: UUID | None = None
  config: dict[str, Any] | None = None


class EvaluationRunConfig(BaseModel):
  llm_judge_provider: str | None = None
  llm_judge_model: str | None = None
  orbitals_claim_extractor_model: str | None = None
  max_concurrent_samples: int | None = None
  max_concurrent_tasks: int | None = None
  sample_trace_timeout_seconds: float | None = None
  sample_compute_timeout_seconds: float | None = None


class EvaluationRunSample(BaseModel):
  """Represents the run of the agent on a single sample in an evaluation run.

  Attributes:
      - id: The unique identifier of the evaluation run sample.
      - evaluation_run_id: The id of the evaluation run this sample belongs to.
      - sample_id: The id of the sample this run belongs to.
      - trace_id: The trace id for this sample's execution (external trace-source ID).
      - status: Execution status for the sample.
      - started_at: Timestamp when sample execution started.
      - ended_at: Timestamp when sample execution ended.
      - error_message: Failure reason for failed samples.
      - metadata: Additional structured sample metadata.
  """

  id: UUID
  evaluation_run_id: UUID
  sample_id: UUID
  trace_id: str | None = None
  status: EvaluationSampleStatus = EvaluationSampleStatus.RUNNING
  started_at: datetime | None = None
  ended_at: datetime | None = None
  error_message: str | None = None
  metadata: dict[str, Any] | None = None


class Trace(BaseModel):
  """Represents a single trace produced by an Agent when processing a sample.
  The trace is the full end-to-end record of the agent’s behavior for a task.
  The trace is a collection of spans.

  Attributes:
     - external_id: The unique identifier of the trace (external trace-source ID).
     - start_time: The start time of the trace.
     - end_time: The end time of the trace.
  """

  external_id: str
  start_time: datetime
  end_time: datetime
  schema_version: int = 1
  source: str = 'unknown'
  adapter: str | None = None


class Span(BaseModel):
  """Represents a single span in a trace. A span is a single, bounded unit of work within an agent’s execution trace.

  Attributes:
     - external_id: The unique identifier of the span (external trace-source ID).
     - trace_id: The id of the trace this span belongs to.
     - parent_span_id: The id of the parent span (nullable for root spans).
     - span_type: The span type.
     - name: The name of the span.
     - start_time: The start time of the span.
     - end_time: The end time of the span.
     - input_data: The input of the span.
     - output_data: The output of the span.
     - metadata: Additional metadata for the span, e.g., LLM input/output messages and tool calls (nullable).
  """

  external_id: str
  trace_id: str
  parent_span_id: str | None = None
  span_type: str
  name: str
  start_time: datetime
  end_time: datetime
  input_data: JsonValue
  output_data: JsonValue
  status: Literal['success', 'error', 'unset'] = 'unset'
  semantics: SpanSemantics = Field(default_factory=SpanSemantics)
  metadata: dict[str, Any] | None = None


class SpanType(BaseModel):
  """Represents a single span type.

  Attributes:
     - name: The name of the span type.
     - description: The description of the span type (nullable).
  """

  name: str
  description: str | None = None


class Metric(BaseModel):
  """Represents a single metric.

  Attributes:
     - name: The name of the metric.
     - description: The description of the metric (nullable)
     - requires_ground_truth: Whether this metric requires ground truth to compute.
     - ground_truth_key: The ground-truth key consumed by this metric, if any.
  """

  name: str
  description: str | None = None
  requires_ground_truth: bool = True
  ground_truth_key: str | None = None


class MetricTargetSpanType(BaseModel):
  """Represents the association between a metric and a span type it targets.

  Attributes:
     - id: The unique identifier of the target.
     - metric: The name of the metric that this span type is a target of.
     - span_type: The span type.
     - targeting_mode: Whether the metric targets spans individually or as a group for this span type.
  """

  id: UUID
  metric: str
  span_type: str
  targeting_mode: MetricTargetingMode = MetricTargetingMode.SINGLE


class MetricComputation(BaseModel):
  """Represents a single metric computation and the spans it evaluated.

  Attributes:
     - id: The unique identifier of the metric computation.
     - evaluation_run_sample_id: The id of the evaluation run on a sample that this computation belongs to.
     - metric: The name of the metric that is being computed.
     - ground_truth_id: The ground truth row used by this computation, if any.
     - targeting_mode: Whether the metric targeted one span or a span group.
     - target_span_type: The span type evaluated by the metric computation.
     - span_ids: The ordered span IDs evaluated by the metric computation; empty when a planned target was skipped
       because no span of the target type was present.
     - score: The score of the computation.
     - status: Execution status for the metric computation.
     - reasoning: Explanation for the score (nullable).
     - metadata: Additional metadata for the computation, e.g., error details (nullable).
     - error_message: Failure reason for failed/skipped metric computations.
     - raw_output: Raw structured output from a metric or judge.
  """

  id: UUID
  evaluation_run_sample_id: UUID
  metric: str
  ground_truth_id: UUID | None = None
  targeting_mode: MetricTargetingMode
  target_span_type: str
  span_ids: list[str] = Field(default_factory=list)
  score: float | None
  status: MetricComputationStatus = MetricComputationStatus.COMPLETED
  reasoning: str | None = None
  metadata: dict[str, Any] | None = None
  error_message: str | None = None
  raw_output: dict[str, Any] | None = None
