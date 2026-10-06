"""Synthetic spans, ground truths and a scripted judge shared by the metric tests."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.model import GroundTruth, Span
from syllo_eval.trace_semantics import RetrievalItem, RetrievalResult, SpanSemantics

from benchmarks.common import Claim, claims_value

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
QUESTION = 'Who approved the budget?'
EXPECTED_ANSWER = 'Dana approved the budget in May.'

# A judge output, an error to raise, or a function of the request returning either.
Reply = dict[str, Any] | Exception | Callable[[LlmJudgeRequest], dict[str, Any] | Exception]


class ScriptedJudge:
  """Answers the n-th request with the n-th reply, or every request with one function of the request."""

  def __init__(self, *replies: Reply, output_tokens: int = 7):
    self._replies = list(replies)
    self.output_tokens = output_tokens
    self.requests: list[LlmJudgeRequest] = []

  async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
    self.requests.append(request)
    reply = self._replies[0] if len(self._replies) == 1 and callable(self._replies[0]) else self._replies.pop(0)
    output = reply(request) if callable(reply) else reply
    if isinstance(output, Exception):
      raise output
    return LlmJudgeResponse(
      provider='fake',
      model='judge',
      output=request.response_model.model_validate(output),
      usage={'input_tokens': 100, 'output_tokens': self.output_tokens, 'total_tokens': 100 + self.output_tokens},
    )

  async def aclose(self) -> None:
    pass


def span(
  external_id: str,
  span_type: str,
  *,
  parent: str | None = 'root',
  offset: int = 0,
  name: str | None = None,
  input_data: Any = None,
  output_data: Any = None,
  trace_id: str = 'trace-1',
  **semantics: Any,
) -> Span:
  return Span(
    external_id=external_id,
    trace_id=trace_id,
    parent_span_id=parent,
    span_type=span_type,
    name=name or span_type,
    start_time=START + timedelta(seconds=offset),
    end_time=START + timedelta(seconds=offset + 1),
    input_data=input_data,
    output_data=output_data,
    status='success',
    semantics=SpanSemantics(**semantics),
  )


def search_span(
  external_id: str,
  document_ids: list[str],
  *,
  offset: int = 1,
  stage: Literal['retrieved', 'selected'] = 'selected',
  ranked: bool = True,
  with_content: bool = True,
) -> Span:
  items = [RetrievalItem(id=doc, content=f'Text of {doc}.' if with_content else None) for doc in document_ids]
  return span(
    external_id,
    'retrieval',
    offset=offset,
    name='search',
    input_data={'query': 'budget approval'},
    output_data=[{'id': doc} for doc in document_ids],
    request=QUESTION,
    retrieval=[RetrievalResult(kind='document', stage=stage, items=items, ranked=ranked)],
  )


def truth(key: str = 'expected_output', **value: Any) -> GroundTruth:
  return GroundTruth(id=uuid4(), sample_id=uuid4(), key=key, ground_truth_value=value or {key: EXPECTED_ANSWER})


def claims_truth(*claims: Claim, key: str = 'expected_claims_gold') -> GroundTruth:
  return GroundTruth(id=uuid4(), sample_id=uuid4(), key=key, ground_truth_value=claims_value(list(claims)))


def truths(*ground_truths: GroundTruth) -> dict[str, GroundTruth]:
  """A sample's ground truths as metrics receive them: keyed by their key."""
  return {ground_truth.key: ground_truth for ground_truth in ground_truths}


def precision_judgments(*decisions: tuple[int, str, bool]) -> dict[str, Any]:
  return {
    'judgments': [
      {'rank': rank, 'retrieved_id': doc, 'relevant': relevant, 'reasoning': f'{doc} reasoning'}
      for rank, doc, relevant in decisions
    ]
  }


def recall_judgments(*decisions: tuple[str, bool]) -> dict[str, Any]:
  return {
    'judgments': [
      {'statement': statement, 'attributable': attributable, 'supporting_retrieved_ids': [], 'reasoning': 'why'}
      for statement, attributable in decisions
    ]
  }


def without_judge_keys(metadata: dict[str, Any] | None) -> dict[str, Any]:
  """Result metadata minus the judge usage, which differs between a per-item and a single-call unit by design."""
  return {key: value for key, value in (metadata or {}).items() if not key.startswith('judge_')}
