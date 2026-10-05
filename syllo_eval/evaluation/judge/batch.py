import asyncio
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any

from syllo_eval.evaluation.judge.base import LlmJudgeResponse
from syllo_eval.evaluation.judge.failures import judge_failure_result
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult


async def judge_batch(
  calls: Iterable[Coroutine[Any, Any, LlmJudgeResponse]],
  *,
  prior_responses: Sequence[LlmJudgeResponse] = (),
  max_output_tokens: int | None = None,
) -> tuple[list[LlmJudgeResponse], MetricComputationResult | None]:
  """Run judge calls concurrently. The first failure cancels the others and gives a classified FAILED result."""
  tasks = [asyncio.create_task(call) for call in calls]
  try:
    return list(await asyncio.gather(*tasks)), None
  except BaseException as exc:
    for task in tasks:
      if not task.done():
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if not isinstance(exc, Exception):
      raise
    responses = [task.result() for task in tasks if not task.cancelled() and task.exception() is None]
    return responses, judge_failure_result(exc, [*prior_responses, *responses], max_output_tokens=max_output_tokens)
