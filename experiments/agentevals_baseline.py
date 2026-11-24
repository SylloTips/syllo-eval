"""The AgentEvals baseline: LangChain's AgentEvals 0.0.9, called on each plan unit of the main pass.

AgentEvals evaluates agent trajectories, given as OpenAI-style message lists or read from a LangGraph thread; the
baseline gives it message lists. It has no retrieval or answer metric, so the baseline scores plans only, on τ²-bench.
Its metric subclasses Plan Correctness, the Syllo-eval metric it is compared with, and replaces only how it judges, so
units, skip rules and the reference plan stay those of Plan Correctness:

- ``AgentEvalsTrajectoryAccuracy``: AgentEvals' trajectory LLM-as-judge with its reference prompt,
  ``TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE``, and its default output, a reasoning and a boolean score. An accurate
  trajectory scores 1.0, any other 0.0.

A unit's trajectory holds what Plan Correctness reads: the request as the user's message, then each executed step as an
assistant message calling the step's tool with its input, followed by the tool's result when the trace observed one.
The reference trajectory is the expected plan, one assistant message per step. Arguments, and results that are not
text, are JSON as Plan Correctness renders them, with sorted keys and non-ASCII characters kept.

AgentEvals judges through OpenEvals, so its evaluator calls the shared judge model as OpenEvals does, through
``openevals_baseline.OpenEvalsJudge``.
"""

import json
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE, create_async_trajectory_llm_as_judge
from pydantic import JsonValue

from syllo_eval.evaluation.metric_support.agent_outputs import extract_actual_plan_steps, non_empty_text
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span

from baseline_judge import NO_SYLLO_EVAL_JUDGE
from openevals_baseline import OpenEvalsJudge, judged, verdict_result

AGENTEVALS_VERSION = version('agentevals')


def agent_trajectory(span: Span) -> list[dict[str, Any]]:
  """The unit's trajectory as OpenAI-style messages: the request, then each executed step's tool call and result."""
  messages: list[dict[str, Any]] = []
  if request := non_empty_text(span.semantics.request):
    messages.append({'role': 'user', 'content': request})
  for step in extract_actual_plan_steps(span):
    tool_call_id = step.attributes.get('tool_call_id')
    call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else step.id
    messages.append(_tool_call(step.operation, step.input, content=step.instruction, call_id=call_id))
    if step.output is not None:
      output = step.output if isinstance(step.output, str) else _json(step.output)
      messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': output})
  return messages


def reference_trajectory(ground_truth: GroundTruth | None) -> list[dict[str, Any]]:
  """The expected plan as OpenAI-style messages: one assistant message calling each step's tool."""
  steps = ground_truth.ground_truth_value.get('expected_plan') if ground_truth is not None else None
  return [
    _tool_call(step['operation'], step.get('parameters', {}), content=step.get('instruction') or '')
    for step in steps or ()
  ]


def _tool_call(operation: str, arguments: JsonValue, *, content: str, call_id: str | None = None) -> dict[str, Any]:
  # AgentEvals reads a tool call's arguments as a JSON object; anything else would be guessed into one.
  if not isinstance(arguments, dict):
    raise ValueError(f'AgentEvals takes tool-call arguments as a JSON object; {operation} has {arguments!r}.')
  call: dict[str, Any] = {'type': 'function', 'function': {'name': operation, 'arguments': _json(arguments)}}
  if call_id is not None:
    call['id'] = call_id
  return {'role': 'assistant', 'content': content, 'tool_calls': [call]}


def _json(value: JsonValue) -> str:
  """JSON as Plan Correctness renders it; LangChain keeps its key order and characters when it re-serializes it."""
  return json.dumps(value, ensure_ascii=False, sort_keys=True)


class AgentEvalsTrajectoryAccuracy(PlanCorrectnessJudgeMetric):
  """AgentEvals' trajectory accuracy against the expected plan, in one call: 1.0 when the judge finds it accurate."""

  metric_name = 'agentevals_trajectory_accuracy'
  metric_description = f'AgentEvals {AGENTEVALS_VERSION} trajectory accuracy against the expected plan, in one call.'

  def __init__(self, *, judge: OpenEvalsJudge, output_token_limit: int):
    super().__init__(judge_client=NO_SYLLO_EVAL_JUDGE)
    self._evaluator = create_async_trajectory_llm_as_judge(
      prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE, judge=judge
    )
    self._output_token_limit = output_token_limit

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    call = judged(
      self._evaluator,
      outputs=agent_trajectory(span),
      reference_outputs=reference_trajectory(ground_truths.get(self.ground_truth_keys[0])),
    )
    return await verdict_result(call, output_token_limit=self._output_token_limit)
