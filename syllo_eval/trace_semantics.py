from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator


class Answer(BaseModel):
  text: str
  kind: Literal['response', 'clarification'] = 'response'
  citations: list[str] = Field(default_factory=list)


class RetrievalItem(BaseModel):
  id: str = Field(min_length=1)
  document_id: str | None = None
  content: str | None = None
  title: str | None = None
  location: str | None = None
  score: float | None = None
  attributes: dict[str, JsonValue] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
  """Items are in observed order, a relevance rank only when ``ranked``; scores are not comparable across searches."""

  kind: str = Field(min_length=1)
  stage: Literal['retrieved', 'reranked', 'selected', 'generation_context'] = 'retrieved'
  query: str | None = None
  availability: Literal['available', 'not_attempted', 'unavailable'] = 'available'
  reason: str | None = None
  items: list[RetrievalItem] = Field(default_factory=list)
  ranked: bool = True

  @model_validator(mode='after')
  def validate_availability(self) -> 'RetrievalResult':
    if self.availability != 'available' and self.items:
      raise ValueError('Unavailable or unattempted retrieval cannot contain items')
    return self


class PlannedStep(BaseModel):
  id: str = Field(min_length=1)
  operation: str = Field(min_length=1)
  instruction: str = ''
  depends_on: list[str] = Field(default_factory=list)
  parameters: dict[str, JsonValue] = Field(default_factory=dict)


class PlanSnapshot(BaseModel):
  id: str = Field(min_length=1)
  supersedes_plan_id: str | None = None
  created_at: datetime | None = None
  steps: list[PlannedStep]

  @model_validator(mode='after')
  def validate_steps(self) -> 'PlanSnapshot':
    ids = {step.id for step in self.steps}
    if len(ids) != len(self.steps):
      raise ValueError('Plan step IDs must be unique')
    pending = {step.id: set(step.depends_on) for step in self.steps}
    if any(not dependencies <= ids for dependencies in pending.values()):
      raise ValueError('Plan dependencies must reference steps in the same plan')
    while pending:
      ready = {key for key, dependencies in pending.items() if not dependencies}
      if not ready:
        raise ValueError('Plan dependencies contain a cycle')
      pending = {key: dependencies - ready for key, dependencies in pending.items() if key not in ready}
    return self


class ExecutionStep(BaseModel):
  id: str = Field(min_length=1)
  operation: str = Field(min_length=1)
  instruction: str = ''
  plan_id: str | None = None
  planned_step_id: str | None = None
  status: Literal['completed', 'unsuccessful', 'error', 'running', 'unknown'] = 'unknown'
  input: JsonValue = None
  output: JsonValue = None
  start_time: datetime | None = None
  end_time: datetime | None = None
  span_ids: list[str] = Field(default_factory=list)
  attributes: dict[str, JsonValue] = Field(default_factory=dict)


class PlanningData(BaseModel):
  plans: list[PlanSnapshot] = Field(default_factory=list)
  executed_steps: list[ExecutionStep] | None = None


class LlmUsage(BaseModel):
  """Own-call usage only. Cache/reasoning counts are details, not additional totals."""

  call_id: str | None = None
  provider: str | None = None
  model: str | None = None
  input_tokens: int | None = Field(default=None, ge=0)
  output_tokens: int | None = Field(default=None, ge=0)
  total_tokens: int | None = Field(default=None, ge=0)
  cached_input_tokens: int | None = Field(default=None, ge=0)
  reasoning_tokens: int | None = Field(default=None, ge=0)
  cost: float | None = Field(default=None, ge=0)
  currency: str | None = None


class SpanSemantics(BaseModel):
  request: str | None = None
  answer: Answer | None = None
  retrieval: list[RetrievalResult] = Field(default_factory=list)
  planning: PlanningData | None = None
  usage: LlmUsage | None = None
