"""The DeepEval baseline of Section 5.1: DeepEval 4.2.6 in test-case mode, on the units of the main pass.

Each metric subclasses the Syllo-eval metric it is compared with and replaces only how it judges, so units, skip rules
and result metadata stay those of the compared metric, and so does scoring for retrieval:

- ``DeepEvalContextualPrecision``: DeepEval's ``ContextualPrecisionMetric`` judges all the documents of a ranking in one
  call. Its verdicts are matched to the documents by position.
- ``DeepEvalGoldClaimsContextualRecall``: DeepEval's ``ContextualRecallMetric`` judges all the ERB gold claims of a unit
  in one call, given one claim per line as the expected output. Its verdicts are matched to the claims by position.
- ``DeepEvalAnswerCorrectness``: DeepEval's G-Eval grades the answer with the Correctness metric of DeepEval's
  documentation, and its score is the result.

The DeepEval metrics keep their own prompts. They call the shared judge model through ``DeepEvalJudge``, which records
every call as a judge response, so that run reports count DeepEval's calls and tokens as they count Syllo-eval's, and
failed units record the same failure classes.
"""

import deepeval_env  # noqa: F401  DeepEval reads these switches on import, so this import comes first.

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from deepeval._version import __version__ as DEEPEVAL_VERSION
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, GEval
from deepeval.metrics.base_metric import Verdict
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.judge.failures import judge_failure_result, mismatch_kind
from syllo_eval.evaluation.metric_support.agent_outputs import display_text, extract_final_answer
from syllo_eval.evaluation.metric_support.retrieved_context import RetrievedItem
from syllo_eval.evaluation.metrics.contracts import EvaluationMetric, MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionJudgment,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import ContextualRecallJudgment
from syllo_eval.infrastructure.exceptions import DataMappingError, ExternalServiceError, JudgeOutputError
from syllo_eval.infrastructure.llm_judge.base import LangChainLlmJudgeClient
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span
from syllo_eval.settings import GeminiJudgeSettings

from ablation_metrics import SEARCH_SPAN_TYPE, GoldClaimsContextualRecall
from config import JudgeConfig
from judge import build_chat_model

PROVIDER = 'gemini'
# The baseline's metrics, keyed by the names the CLI selects them with.
METRIC_KEYS = ('precision', 'recall', 'answer')

# The Correctness metric of DeepEval's G-Eval documentation, verbatim. With evaluation steps, the criteria only
# describe the metric: DeepEval generates no steps from them.
CORRECTNESS_CRITERIA = 'Determine whether the actual output is factually correct based on the expected output.'
CORRECTNESS_STEPS = (
  "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
  'You should also heavily penalize omission of detail',
  'Vague language, or contradicting OPINIONS, are OK',
)

_recorded_calls: ContextVar[list[LlmJudgeResponse]] = ContextVar('deepeval_judge_calls')


@contextmanager
def recorded_judge_calls() -> Iterator[list[LlmJudgeResponse]]:
  """Collect the judge calls that DeepEval makes in this context, so each unit reports its own calls."""
  calls: list[LlmJudgeResponse] = []
  token = _recorded_calls.set(calls)
  try:
    yield calls
  finally:
    _recorded_calls.reset(token)


class DeepEvalJudge(DeepEvalBaseLLM):
  """The shared judge model, called the way DeepEval calls a custom model.

  A DeepEval prompt goes out as the only message, with the JSON schema of its output and no output limit, as in
  DeepEval's own Gemini model. Everything else mirrors Syllo-eval's judge client: the temperature, the concurrency limit
  and the retries of rate limits come from the experiment config. Usage is read and the output validated the same way,
  so an output that does not parse or validate raises with its usage and never reaches DeepEval's lenient JSON parsing.
  Every call must run inside ``recorded_judge_calls``, which receives the successful ones.
  """

  def __init__(
    self,
    chat_model: ChatGoogleGenerativeAI,
    *,
    temperature: float,
    max_concurrent_requests: int,
    max_attempts: int,
  ):
    self._chat_model = chat_model.model_copy(update={'temperature': temperature})
    self._semaphore = asyncio.Semaphore(max_concurrent_requests)
    self._max_attempts = max_attempts
    super().__init__(model=chat_model.model)

  def load_model(self, *args: Any, **kwargs: Any) -> Any:
    return self._chat_model

  def get_model_name(self, *args: Any, **kwargs: Any) -> str:
    return self._chat_model.model

  def generate(self, *args: Any, **kwargs: Any) -> str:
    raise NotImplementedError('The baseline runs DeepEval metrics in async mode only.')

  # DeepEval types the generation methods as taking anything and returning text, but its metrics call them with a
  # prompt and a schema, and take a schema instance back as the parsed output.
  async def a_generate_with_schema(  # type: ignore[override]
    self, prompt: str, schema: type[BaseModel] | None = None
  ) -> BaseModel:
    # DeepEval's default calls again without the schema when this raises TypeError, which would hide an error here.
    return await self.a_generate(prompt, schema)

  async def a_generate(  # type: ignore[override]
    self, prompt: str, schema: type[BaseModel] | None = None
  ) -> BaseModel:
    if schema is None:
      raise ValueError('The baseline metrics always ask the judge for structured output.')
    calls = _recorded_calls.get()
    runnable = self._chat_model.with_structured_output(
      schema=schema.model_json_schema(), method='json_schema', include_raw=True
    )
    async with self._semaphore:
      result, latency_seconds, attempts = await self._invoke(runnable, prompt)

    raw = result['raw']
    if not isinstance(raw, AIMessage):
      raise DataMappingError(PROVIDER, 'structured output runnable did not return an AI message')
    # A response that cannot be used still cost its tokens.
    usage = LangChainLlmJudgeClient._usage(raw)
    if result['parsing_error'] is not None:
      raise JudgeOutputError(PROVIDER, 'failed to parse structured output', result['parsing_error'], usage=usage)
    if result['parsed'] is None:
      refusal = raw.additional_kwargs.get('refusal')
      reason = f'model refused judge request: {refusal}' if refusal else 'model returned no structured output'
      raise JudgeOutputError(PROVIDER, reason, usage=usage)
    try:
      output = schema.model_validate(result['parsed'])
    except ValidationError as error:
      raise JudgeOutputError(PROVIDER, 'invalid structured output', error, usage=usage) from error

    calls.append(
      LlmJudgeResponse(
        provider=PROVIDER,
        model=raw.response_metadata.get('model_name') or self._chat_model.model,
        output=output,
        response_id=(
          raw.id
          or raw.response_metadata.get('id')
          or raw.additional_kwargs.get('response_id')
          or raw.additional_kwargs.get('id')
        ),
        usage=usage,
        latency_seconds=latency_seconds,
        attempts=attempts,
      )
    )
    return output

  async def _invoke(self, runnable: Any, prompt: str) -> tuple[dict[str, Any], float, int]:
    """The structured-output result, the latency of the attempt that succeeded, and the number of attempts."""
    attempt = 1
    while True:
      started = time.monotonic()
      try:
        return await runnable.ainvoke([('human', prompt)]), time.monotonic() - started, attempt
      except Exception as error:
        failure = ExternalServiceError(PROVIDER, 'judge', error)
        retry_delay = GeminiLlmJudgeClient._retry_delay_seconds(failure)
        if retry_delay is None or attempt >= self._max_attempts:
          raise failure from error
        attempt += 1
        await asyncio.sleep(retry_delay)


class _NoSylloEvalJudge:
  """The judge client of the compared metrics' constructors: the baseline metrics judge through DeepEval only."""

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    raise RuntimeError('DeepEval baseline metrics judge through DeepEval, not through a Syllo-eval judge client.')

  async def aclose(self) -> None:
    pass


_NO_SYLLO_EVAL_JUDGE = _NoSylloEvalJudge()


def node_text(item: RetrievedItem) -> str:
  """A retrieved document as a DeepEval retrieval-context node: its title, if any, then its content."""
  return '\n\n'.join(part for part in (item.item.title, item.item.content) if part)


class DeepEvalContextualPrecision(ContextualPrecisionDocumentJudgeMetric):
  """DeepEval's contextual precision: all the documents of a ranking judged in one call, matched by position."""

  metric_name = 'deepeval_contextual_precision'
  metric_description = f'DeepEval {DEEPEVAL_VERSION} contextual precision, judging a ranking in one call.'

  def __init__(
    self, *, judge: DeepEvalJudge, output_token_limit: int, target_span_types: Sequence[str] = ('agent_root',)
  ):
    super().__init__(judge_client=_NO_SYLLO_EVAL_JUDGE, target_span_types=target_span_types)
    self._judge = judge
    self._output_token_limit = output_token_limit

  async def judge_items(
    self, span: Span, ground_truth: GroundTruth, rankings: list[list[RetrievedItem]]
  ) -> tuple[list[LlmJudgeResponse], list[ContextualPrecisionJudgment]] | MetricComputationResult:
    judgments: list[ContextualPrecisionJudgment] = []
    with recorded_judge_calls() as calls:
      for items in rankings:
        if not items:
          # An observed empty ranking needs no call.
          continue
        metric = ContextualPrecisionMetric(model=self._judge, include_reason=False, async_mode=True, eval_mode='llm')
        test_case = LLMTestCase(
          input=span.semantics.request or '<empty>',
          expected_output=ground_truth.ground_truth_value['expected_output'],
          retrieval_context=[node_text(item) for item in items],
        )
        try:
          await metric.a_measure(test_case, _show_indicator=False)
        except (DataMappingError, ExternalServiceError) as error:
          return judge_failure_result(error, calls, max_output_tokens=self._output_token_limit)
        if len(metric.verdicts) != len(items):
          message = (
            f'DeepEval returned {len(metric.verdicts)} contextual precision verdicts for {len(items)} ranked '
            f'{self.variant}s.'
          )
          return _misaligned_result(
            self.name, message, calls, self._output_token_limit, variant=self.variant, retrieval_kind=self.variant
          )
        judgments.extend(
          ContextualPrecisionJudgment(
            rank=rank, retrieved_id=item.retrieved_id, relevant=verdict.verdict == Verdict.YES, reasoning=verdict.reason
          )
          for rank, (item, verdict) in enumerate(zip(items, metric.verdicts), start=1)
        )
    return calls, judgments


class DeepEvalGoldClaimsContextualRecall(GoldClaimsContextualRecall):
  """DeepEval's contextual recall over the ERB gold claims: all claims judged in one call, matched by position."""

  metric_name = 'deepeval_contextual_recall_gold_claims'
  metric_description = f'DeepEval {DEEPEVAL_VERSION} contextual recall over the ERB gold claims, in one call.'

  def __init__(
    self, *, judge: DeepEvalJudge, output_token_limit: int, target_span_types: Sequence[str] = ('agent_root',)
  ):
    super().__init__(judge_client=_NO_SYLLO_EVAL_JUDGE, target_span_types=target_span_types)
    self._judge = judge
    self._output_token_limit = output_token_limit

  async def judge_claims(
    self,
    span: Span,
    retrieved_items: list[RetrievedItem],
    claims: list[str],
    *,
    prior_responses: Sequence[LlmJudgeResponse] = (),
  ) -> tuple[list[LlmJudgeResponse], list[ContextualRecallJudgment]] | MetricComputationResult:
    # DeepEval judges each sentence of the expected output: one claim per line keeps one claim per sentence.
    if any('\n' in claim for claim in claims):
      raise ValueError('DeepEval recall gives one claim per line, so a claim cannot span several lines.')
    metric = ContextualRecallMetric(model=self._judge, include_reason=False, async_mode=True, eval_mode='llm')
    test_case = LLMTestCase(
      input=span.semantics.request or '<empty>',
      expected_output='\n'.join(claims),
      retrieval_context=[node_text(item) for item in retrieved_items],
    )
    with recorded_judge_calls() as calls:
      try:
        await metric.a_measure(test_case, _show_indicator=False)
      except (DataMappingError, ExternalServiceError) as error:
        return judge_failure_result(error, [*prior_responses, *calls], max_output_tokens=self._output_token_limit)
    if len(metric.verdicts) != len(claims):
      message = f'DeepEval returned {len(metric.verdicts)} contextual recall verdicts for {len(claims)} statements.'
      return _misaligned_result(
        self.name,
        message,
        [*prior_responses, *calls],
        self._output_token_limit,
        variant=self.variant,
        retrieval_kind=self.variant,
      )
    judgments = [
      ContextualRecallJudgment(
        statement=claim,
        attributable=verdict.verdict == Verdict.YES,
        supporting_retrieved_ids=[],
        reasoning=verdict.reason,
      )
      for claim, verdict in zip(claims, metric.verdicts)
    ]
    return calls, judgments


class DeepEvalAnswerCorrectness(AnswerCorrectnessJudgeMetric):
  """DeepEval's G-Eval with the Correctness metric of its documentation; the score is G-Eval's 0-10 score over 10."""

  metric_name = 'deepeval_answer_correctness'
  metric_description = f'DeepEval {DEEPEVAL_VERSION} G-Eval Correctness of the answer against the expected one.'

  def __init__(self, *, judge: DeepEvalJudge, output_token_limit: int):
    super().__init__(judge_client=_NO_SYLLO_EVAL_JUDGE)
    self._judge = judge
    self._output_token_limit = output_token_limit

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    metric = GEval(
      name='Correctness',
      criteria=CORRECTNESS_CRITERIA,
      evaluation_steps=list(CORRECTNESS_STEPS),
      evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
      model=self._judge,
      async_mode=True,
    )
    test_case = LLMTestCase(
      input=display_text(span.semantics.request),
      actual_output=extract_final_answer(span),
      expected_output=self._extract_expected_answer(ground_truths.get(self.ground_truth_keys[0])),
    )
    with recorded_judge_calls() as calls:
      try:
        await metric.a_measure(test_case, _show_indicator=False)
      except (DataMappingError, ExternalServiceError) as error:
        return judge_failure_result(error, calls, max_output_tokens=self._output_token_limit)
    return MetricComputationResult(score=metric.score, reasoning=metric.reason, metadata=judge_metadata(calls))


def _misaligned_result(
  metric_name: str,
  message: str,
  responses: Sequence[LlmJudgeResponse],
  output_token_limit: int,
  **metadata: Any,
) -> MetricComputationResult:
  """A unit whose verdicts, from its last call, do not match its items: misaligned, or truncated at the output limit."""
  response = responses[-1]
  return MetricComputationResult(
    score=None,
    status=MetricComputationStatus.FAILED,
    reasoning=f'Failed to compute {metric_name}: {message}',
    metadata={**metadata, 'failure': mismatch_kind(response, output_token_limit).value, **judge_metadata(responses)},
    error_message=message,
    raw_output=response.output.model_dump(mode='json'),
  )


def deepeval_metrics(judge: DeepEvalJudge, *, output_token_limit: int) -> dict[str, EvaluationMetric]:
  """The baseline's metrics by key. The retrieval metrics score search units, as in the main pass."""
  return {
    'precision': DeepEvalContextualPrecision(
      judge=judge, output_token_limit=output_token_limit, target_span_types=(SEARCH_SPAN_TYPE,)
    ),
    'recall': DeepEvalGoldClaimsContextualRecall(
      judge=judge, output_token_limit=output_token_limit, target_span_types=(SEARCH_SPAN_TYPE,)
    ),
    'answer': DeepEvalAnswerCorrectness(judge=judge, output_token_limit=output_token_limit),
  }


@asynccontextmanager
async def open_deepeval_judge(config: JudgeConfig, settings: GeminiJudgeSettings) -> AsyncIterator[DeepEvalJudge]:
  """Yield the shared judge as DeepEval calls it: the model, temperature, concurrency and retries of ``config``."""
  chat_model = build_chat_model(config, settings)
  try:
    yield DeepEvalJudge(
      chat_model,
      temperature=config.temperature,
      max_concurrent_requests=config.max_concurrent_requests,
      max_attempts=config.max_retries,
    )
  finally:
    await chat_model.async_client.aclose()
