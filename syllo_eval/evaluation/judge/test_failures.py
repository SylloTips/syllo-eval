import unittest

import httpx
from langchain_core.exceptions import ContextOverflowError

from syllo_eval.evaluation.judge.base import LlmJudgeResponse
from syllo_eval.evaluation.judge.batch import judge_batch
from syllo_eval.evaluation.judge.failures import JudgeFailureKind, classify_judge_error
from syllo_eval.evaluation.judge.metric_base import JudgeScorePayload
from syllo_eval.infrastructure.exceptions import DataMappingError, ExternalServiceError, JudgeOutputError
from syllo_eval.model import MetricComputationStatus


class _GatewayTimeout(Exception):
  code = 504


class ClassifyJudgeErrorTest(unittest.TestCase):
  def test_failure_kinds(self) -> None:
    cases: list[tuple[Exception, int | None, JudgeFailureKind]] = [
      (DataMappingError('judge', 'no structured output'), None, JudgeFailureKind.INVALID_OUTPUT),
      (JudgeOutputError('judge', 'invalid', usage={'output_tokens': 99}), 100, JudgeFailureKind.INVALID_OUTPUT),
      (JudgeOutputError('judge', 'invalid', usage={'output_tokens': 100}), 100, JudgeFailureKind.TRUNCATED),
      (
        ExternalServiceError('judge', 'judge', ContextOverflowError('too long')),
        None,
        JudgeFailureKind.CONTEXT_OVERFLOW,
      ),
      (ExternalServiceError('judge', 'judge', httpx.ReadTimeout('slow')), None, JudgeFailureKind.TIMEOUT),
      (ExternalServiceError('judge', 'judge', _GatewayTimeout('deadline exceeded')), None, JudgeFailureKind.TIMEOUT),
      (ExternalServiceError('judge', 'judge', RuntimeError('500 internal')), None, JudgeFailureKind.PROVIDER),
    ]
    for error, max_output_tokens, kind in cases:
      with self.subTest(error=str(error), max_output_tokens=max_output_tokens):
        self.assertEqual(classify_judge_error(error, max_output_tokens), kind)

  def test_follows_the_exception_chain(self) -> None:
    try:
      try:
        raise ContextOverflowError('too long')
      except ContextOverflowError as cause:
        raise ExternalServiceError('judge', 'judge') from cause
    except ExternalServiceError as error:
      self.assertEqual(classify_judge_error(error), JudgeFailureKind.CONTEXT_OVERFLOW)


class JudgeBatchFailureTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_failure_is_classified_and_counts_every_call_with_its_usage(self) -> None:
    async def judged() -> LlmJudgeResponse:
      output = JudgeScorePayload(score=1, reasoning='ok')
      return LlmJudgeResponse(provider='fake', model='judge', output=output, usage={'input_tokens': 3})

    async def cut_off() -> LlmJudgeResponse:
      raise JudgeOutputError('fake', 'invalid structured output', usage={'input_tokens': 5, 'output_tokens': 10})

    responses, failure = await judge_batch([judged(), cut_off()], max_output_tokens=10)

    self.assertEqual(len(responses), 1)
    assert failure is not None and failure.metadata is not None
    self.assertEqual(failure.status, MetricComputationStatus.FAILED)
    self.assertEqual(failure.metadata['failure'], 'truncated')
    self.assertEqual((failure.metadata['judge_calls'], failure.metadata['judge_failed_calls']), (2, 1))
    self.assertEqual(failure.metadata['judge_usage'], {'input_tokens': 8, 'output_tokens': 10})


if __name__ == '__main__':
  unittest.main()
