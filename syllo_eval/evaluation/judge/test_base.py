import unittest

from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeResponse, judge_metadata


class _Payload(BaseModel):
  value: int


class JudgeMetadataTest(unittest.TestCase):
  def test_judge_metadata_uses_single_response_id_or_aggregated_response_ids(self) -> None:
    first = LlmJudgeResponse(
      provider='fake',
      model='judge-model',
      output=_Payload(value=1),
      response_id='response-1',
      usage={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
    )
    second = LlmJudgeResponse(
      provider='fake',
      model='judge-model',
      output=_Payload(value=2),
      response_id='response-2',
      usage={'input_tokens': 4, 'output_tokens': 1, 'total_tokens': 5},
    )

    self.assertEqual(
      judge_metadata([first]),
      {
        'judge_provider': 'fake',
        'judge_model': 'judge-model',
        'judge_calls': 1,
        'judge_attempts': 1,
        'judge_response_id': 'response-1',
        'judge_usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
      },
    )
    self.assertEqual(
      judge_metadata([first, second]),
      {
        'judge_provider': 'fake',
        'judge_model': 'judge-model',
        'judge_calls': 2,
        'judge_attempts': 2,
        'judge_response_ids': ['response-1', 'response-2'],
        'judge_usage': {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10},
      },
    )

  def test_judge_metadata_sums_latency_attempts_and_reported_detail_counts(self) -> None:
    retried = LlmJudgeResponse(
      provider='fake',
      model='judge-model',
      output=_Payload(value=1),
      usage={'input_tokens': 30, 'output_tokens': 2, 'total_tokens': 32, 'cached_input_tokens': 20},
      latency_seconds=1.25,
      attempts=2,
    )
    first_try = LlmJudgeResponse(
      provider='fake',
      model='judge-model',
      output=_Payload(value=2),
      usage={'input_tokens': 10, 'output_tokens': 1, 'total_tokens': 11},
      latency_seconds=0.5,
    )

    metadata = judge_metadata([retried, first_try])

    self.assertEqual(metadata['judge_calls'], 2)
    self.assertEqual(metadata['judge_attempts'], 3)
    self.assertEqual(metadata['judge_latency_seconds'], 1.75)
    self.assertEqual(
      metadata['judge_usage'],
      {'input_tokens': 40, 'output_tokens': 3, 'total_tokens': 43, 'cached_input_tokens': 20},
    )

  def test_judge_metadata_omits_latency_when_no_response_reports_it(self) -> None:
    response = LlmJudgeResponse(provider='fake', model='judge-model', output=_Payload(value=1))

    self.assertNotIn('judge_latency_seconds', judge_metadata([response]))


if __name__ == '__main__':
  unittest.main()
