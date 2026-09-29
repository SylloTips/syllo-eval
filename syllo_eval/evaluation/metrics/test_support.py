"""Small provider-neutral inputs shared by metric contract tests."""

from datetime import datetime, timezone
from uuid import uuid4
from syllo_eval.evaluation.judge import LlmJudgeResponse
from syllo_eval.model import GroundTruth, Span
from syllo_eval.trace_semantics import SpanSemantics


def span(**semantics) -> Span:
  now = datetime.now(timezone.utc)
  return Span(
    external_id='s',
    trace_id='t',
    span_type='retrieval',
    name='search',
    start_time=now,
    end_time=now,
    input_data=None,
    output_data=None,
    semantics=SpanSemantics(request='Question', **semantics),
  )


def truth(**payload) -> GroundTruth:
  return GroundTruth(id=uuid4(), sample_id=uuid4(), key='expected_output', ground_truth_value=payload)


class Judge:
  def __init__(self, *outputs):
    self.outputs = iter(outputs)
    self.requests = []

  async def judge(self, request):
    self.requests.append(request)
    return LlmJudgeResponse(
      provider='fake',
      model='test',
      output=request.response_model.model_validate(next(self.outputs)),
      usage={'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3},
    )

  async def aclose(self):
    pass
