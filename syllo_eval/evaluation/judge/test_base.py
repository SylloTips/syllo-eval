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
        'judge_response_id': 'response-1',
        'judge_usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
      },
    )
    self.assertEqual(
      judge_metadata([first, second]),
      {
        'judge_provider': 'fake',
        'judge_model': 'judge-model',
        'judge_response_ids': ['response-1', 'response-2'],
        'judge_usage': {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10},
      },
    )


if __name__ == '__main__':
  unittest.main()
