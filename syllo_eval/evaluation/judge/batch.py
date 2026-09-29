import asyncio
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any

from syllo_eval.evaluation.judge.base import LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.model import MetricComputationStatus


async def judge_batch(
  calls: Iterable[Coroutine[Any, Any, LlmJudgeResponse]],
  *,
  prior_responses: Sequence[LlmJudgeResponse] = (),
) -> tuple[list[LlmJudgeResponse], MetricComputationResult | None]:
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
    return responses, MetricComputationResult(
      score=None,
      status=MetricComputationStatus.FAILED,
      error_message=str(exc),
      metadata=judge_metadata([*prior_responses, *responses]),
    )
