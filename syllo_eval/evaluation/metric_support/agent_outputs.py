import json
from collections.abc import Sequence

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span
from syllo_eval.trace_semantics import ExecutionStep


def ground_truth_text(ground_truth: GroundTruth | None, key: str) -> str | None:
  if ground_truth is None:
    return None

  value = ground_truth.ground_truth_value.get(key)
  if isinstance(value, list):
    return render_plan(value)
  return non_empty_text(value)


def non_empty_text(value: object) -> str | None:
  if not isinstance(value, str):
    return None

  stripped = value.strip()
  return stripped or None


def display_text(value: object) -> str:
  return non_empty_text(value) or '<empty>'


def extract_final_answer(span: Span) -> str:
  answer = span.semantics.answer
  return display_text(answer.text) if answer is not None else '<empty>'


def extract_actual_plan(span: Span) -> str:
  return render_plan(extract_actual_plan_steps(span))


def extract_actual_plan_steps(span: Span) -> list[ExecutionStep]:
  planning = span.semantics.planning
  if planning is None or planning.executed_steps is None:
    return []

  return planning.executed_steps


def render_plan(steps: Sequence[object]) -> str:
  rendered_steps = [
    rendered for index, step in enumerate(steps, start=1) if (rendered := _render_step(index, step)) is not None
  ]
  return '\n'.join(rendered_steps) if rendered_steps else '<empty>'


def _render_step(index: int, step: object) -> str | None:
  if isinstance(step, ExecutionStep):
    # Defaults mean not observed: an 'unknown' status, empty instruction, or missing input or output.
    step = step.model_dump(exclude_defaults=True)
  elif not isinstance(step, dict):
    return None

  operation_text = non_empty_text(step.get('operation')) or '<unknown operation>'
  parts = [f'Step {index}: operation={operation_text}']
  for name in ('status', 'instruction', 'input', 'parameters', 'output'):
    text = _render_step_field(step.get(name))
    if text is not None:
      parts.append(f'  {name}: {text}')
  return '\n'.join(parts)


def _render_step_field(value: object) -> str | None:
  if value is None:
    return None
  if isinstance(value, str):
    return non_empty_text(value)
  return json.dumps(value, ensure_ascii=False, sort_keys=True)


def skipped_retrieval_result(metric_name: str, reason: str, *, variant: str) -> MetricComputationResult:
  return MetricComputationResult(
    score=None,
    status=MetricComputationStatus.SKIPPED,
    reasoning=f'Skipped {metric_name}: {reason}',
    metadata={'variant': variant},
    error_message=reason,
  )
