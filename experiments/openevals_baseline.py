"""The OpenEvals baseline: LangChain's OpenEvals 0.2.2, called on the units of the main pass.

OpenEvals ships LLM-as-judge prompts that each return one verdict per call. Each metric subclasses the Syllo-eval
metric it is compared with and replaces only how it judges, so units, skip rules and result metadata stay those of the
compared metric, and so does scoring where OpenEvals judges the same items:

- ``OpenEvalsRetrievalRelevance``: ``RAG_RETRIEVAL_RELEVANCE_PROMPT`` on each document of a ranking, one call per
  document; contextual precision scores the verdicts.
- ``OpenEvalsGoldClaimsGroundedness`` and ``OpenEvalsOrbitalsClaimsGroundedness``: OpenEvals has no recall evaluator.
  ``RAG_GROUNDEDNESS_PROMPT`` decides whether an output is supported by the retrieved context; given each claim of a
  unit as the output, one call per claim, it decides the claim's attribution, and contextual recall scores the verdicts.
- ``OpenEvalsCorrectness``: ``CORRECTNESS_PROMPT``, with the expected answer as the reference outputs.
- ``OpenEvalsPlanAdherence``: ``PLAN_ADHERENCE_PROMPT``, with the expected plan as the plan.

Every evaluator keeps OpenEvals' default output, a reasoning and a boolean score; the answer and plan metrics score 1.0
when it is true and 0.0 otherwise. The evaluators call the shared judge model through ``OpenEvalsJudge``, which records
every call as a judge response (``baseline_judge``), so that run reports count OpenEvals' calls and tokens as they count
Syllo-eval's, and failed units record the same failure classes. AgentEvals judges through OpenEvals, so its baseline
uses the same judge (``agentevals_baseline``).
"""

from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
  CORRECTNESS_PROMPT,
  PLAN_ADHERENCE_PROMPT,
  RAG_GROUNDEDNESS_PROMPT,
  RAG_RETRIEVAL_RELEVANCE_PROMPT,
)
from pydantic import BaseModel, ConfigDict, PrivateAttr, StrictBool

from syllo_eval.evaluation.judge import LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.judge.batch import judge_batch
from syllo_eval.evaluation.metric_support.agent_outputs import (
  display_text,
  extract_actual_plan,
  extract_final_answer,
  ground_truth_text,
)
from syllo_eval.evaluation.metric_support.retrieved_context import RetrievedItem
from syllo_eval.evaluation.metrics.contracts import EvaluationMetric, MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionJudgment,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import ContextualRecallJudgment
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span
from syllo_eval.settings import GeminiJudgeSettings

from ablation_metrics import SEARCH_SPAN_TYPE, GoldClaimsContextualRecall
from baseline_judge import NO_SYLLO_EVAL_JUDGE, BaselineJudge, node_text, recorded_judge_calls
from benchmarks.orbitals_claims import CLAIMS_KEY as ORBITALS_CLAIMS_KEY
from config import Benchmark, JudgeConfig
from judge import build_chat_model

OPENEVALS_VERSION = version('openevals')
# The baseline's metrics, keyed by the names the CLI selects them with.
METRIC_KEYS = ('relevance', 'recall', 'answer', 'plan')


class BooleanScore(BaseModel):
  """OpenEvals' default judge output: a reasoning, then whether the prompt's criteria are met."""

  model_config = ConfigDict(extra='forbid')

  reasoning: str
  score: StrictBool


class OpenEvalsJudge(BaseChatModel):
  """The shared judge model, called the way OpenEvals calls a LangChain chat model.

  OpenEvals asks the model for structured output with the JSON schema of its default output, then sends its prompt as
  the only message. Both go to the shared judge model as LangChain sends them to Gemini: with the ``json_schema``
  method, LangChain's default for Gemini, which OpenEvals leaves to the model, and no output limit, as OpenEvals sets
  none. The rest is the shared judge's (``baseline_judge.BaselineJudge``), so an output that does not parse or validate
  raises with its usage. Every call must run inside ``recorded_judge_calls``, which receives the successful ones.
  """

  _judge: BaselineJudge = PrivateAttr()

  def __init__(
    self,
    chat_model: ChatGoogleGenerativeAI,
    *,
    temperature: float,
    max_concurrent_requests: int,
    max_attempts: int,
  ):
    super().__init__()
    self._judge = BaselineJudge(
      chat_model,
      temperature=temperature,
      max_concurrent_requests=max_concurrent_requests,
      max_attempts=max_attempts,
    )

  @property
  def _llm_type(self) -> str:
    return 'openevals-baseline-judge'

  def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
    raise NotImplementedError('The OpenEvals baseline judges through structured output only.')

  def with_structured_output(  # type: ignore[override]
    self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
  ) -> Runnable[Any, dict[str, Any]]:
    # Another schema would make outputs fail validation as the judge's own failures, so it is refused instead.
    if include_raw or kwargs or not _is_boolean_score(schema):
      raise ValueError('The OpenEvals baseline asks only for the default output of OpenEvals judges.')
    runnable = self._judge.chat_model.with_structured_output(schema=schema, method='json_schema', include_raw=True)

    async def judge(messages: list[dict[str, Any]]) -> dict[str, Any]:
      output = await self._judge.generate(runnable, messages, BooleanScore.model_validate)
      return output.model_dump()

    return RunnableLambda(judge)


def _is_boolean_score(schema: object) -> bool:
  """Whether ``schema`` asks for ``BooleanScore``: a reasoning and a boolean score, both required."""
  if not isinstance(schema, dict):
    return False
  properties = schema.get('properties') or {}
  return (
    sorted(properties) == sorted(BooleanScore.model_fields)
    and sorted(schema.get('required') or ()) == sorted(BooleanScore.model_fields)
    and properties['reasoning'].get('type') == 'string'
    and properties['score'].get('type') == 'boolean'
  )


async def judged(evaluator: Any, **values: Any) -> LlmJudgeResponse[BooleanScore]:
  """One call of an OpenEvals evaluator, as the judge response it made: its output is the evaluator's verdict."""
  with recorded_judge_calls() as calls:
    await evaluator(**values)
  [response] = calls
  return response


async def verdict_result(
  call: Coroutine[Any, Any, LlmJudgeResponse[BooleanScore]], *, output_token_limit: int
) -> MetricComputationResult:
  """The result of a unit judged in one call: 1.0 when the verdict is true, else 0.0, or the failure of the call."""
  responses, failure = await judge_batch([call], max_output_tokens=output_token_limit)
  if failure is not None:
    return failure
  [response] = responses
  return MetricComputationResult(
    score=1.0 if response.output.score else 0.0, reasoning=response.output.reasoning, metadata=judge_metadata(responses)
  )


def _documents(items: Sequence[RetrievedItem]) -> dict[str, list[str]]:
  """Retrieved context as OpenEvals' examples give it, which its prompts render with ``str()``."""
  return {'documents': [node_text(item) for item in items]}


class OpenEvalsRetrievalRelevance(ContextualPrecisionDocumentJudgeMetric):
  """OpenEvals' retrieval relevance of each document of a ranking, one call per document, scored as precision."""

  metric_name = 'openevals_retrieval_relevance'
  metric_description = (
    f'OpenEvals {OPENEVALS_VERSION} retrieval relevance of each ranked document, scored as precision.'
  )

  def __init__(
    self, *, judge: OpenEvalsJudge, output_token_limit: int, target_span_types: Sequence[str] = ('agent_root',)
  ):
    super().__init__(judge_client=NO_SYLLO_EVAL_JUDGE, target_span_types=target_span_types)
    self._evaluator = create_async_llm_as_judge(
      prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT, feedback_key='retrieval_relevance', judge=judge
    )
    self._output_token_limit = output_token_limit

  async def judge_items(
    self, span: Span, ground_truth: GroundTruth, rankings: list[list[RetrievedItem]]
  ) -> tuple[list[LlmJudgeResponse], list[ContextualPrecisionJudgment]] | MetricComputationResult:
    ranked_items = [(rank, item) for items in rankings for rank, item in enumerate(items, start=1)]
    request = span.semantics.request or '<empty>'
    responses, failure = await judge_batch(
      (judged(self._evaluator, inputs=request, context=_documents([item])) for _, item in ranked_items),
      max_output_tokens=self._output_token_limit,
    )
    if failure is not None:
      return failure
    return responses, [
      ContextualPrecisionJudgment(
        rank=rank,
        retrieved_id=item.retrieved_id,
        relevant=response.output.score,
        reasoning=response.output.reasoning,
      )
      for (rank, item), response in zip(ranked_items, responses)
    ]


class OpenEvalsGoldClaimsGroundedness(GoldClaimsContextualRecall):
  """OpenEvals' groundedness of each ERB gold claim in the retrieved evidence, one call per claim, scored as recall."""

  metric_name = 'openevals_groundedness_gold_claims'
  metric_description = (
    f'OpenEvals {OPENEVALS_VERSION} groundedness of each ERB gold claim in the retrieved evidence, scored as recall.'
  )

  def __init__(
    self, *, judge: OpenEvalsJudge, output_token_limit: int, target_span_types: Sequence[str] = ('agent_root',)
  ):
    super().__init__(judge_client=NO_SYLLO_EVAL_JUDGE, target_span_types=target_span_types)
    self._evaluator = create_async_llm_as_judge(
      prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key='groundedness', judge=judge
    )
    self._output_token_limit = output_token_limit

  async def judge_claims(
    self,
    span: Span,
    retrieved_items: list[RetrievedItem],
    claims: list[str],
    *,
    prior_responses: Sequence[LlmJudgeResponse] = (),
  ) -> tuple[list[LlmJudgeResponse], list[ContextualRecallJudgment]] | MetricComputationResult:
    context = _documents(retrieved_items)
    responses, failure = await judge_batch(
      (judged(self._evaluator, context=context, outputs=claim) for claim in claims),
      prior_responses=prior_responses,
      max_output_tokens=self._output_token_limit,
    )
    if failure is not None:
      return failure
    return responses, [
      ContextualRecallJudgment(
        statement=claim,
        attributable=response.output.score,
        supporting_retrieved_ids=[],
        reasoning=response.output.reasoning,
      )
      for claim, response in zip(claims, responses)
    ]


class OpenEvalsOrbitalsClaimsGroundedness(OpenEvalsGoldClaimsGroundedness):
  """The same over the Orbitals claims of a WixQA expected answer, as the main pass scores WixQA's recall."""

  claims_key = ORBITALS_CLAIMS_KEY
  metric_name = 'openevals_groundedness_orbitals_claims'
  metric_description = (
    f'OpenEvals {OPENEVALS_VERSION} groundedness of each Orbitals claim of the WixQA expected answer in the retrieved '
    'evidence, scored as recall.'
  )


class OpenEvalsCorrectness(AnswerCorrectnessJudgeMetric):
  """OpenEvals' correctness of the answer against the expected answer, in one call: 1.0 when it is correct."""

  metric_name = 'openevals_correctness'
  metric_description = f'OpenEvals {OPENEVALS_VERSION} correctness of the answer against the expected one.'

  def __init__(self, *, judge: OpenEvalsJudge, output_token_limit: int):
    super().__init__(judge_client=NO_SYLLO_EVAL_JUDGE)
    self._evaluator = create_async_llm_as_judge(prompt=CORRECTNESS_PROMPT, feedback_key='correctness', judge=judge)
    self._output_token_limit = output_token_limit

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    call = judged(
      self._evaluator,
      inputs=display_text(span.semantics.request),
      outputs=extract_final_answer(span),
      reference_outputs=self._extract_expected_answer(ground_truths.get(self.ground_truth_keys[0])),
    )
    return await verdict_result(call, output_token_limit=self._output_token_limit)


class OpenEvalsPlanAdherence(PlanCorrectnessJudgeMetric):
  """OpenEvals' plan adherence of the executed steps to the expected plan, in one call: 1.0 when they follow it."""

  metric_name = 'openevals_plan_adherence'
  metric_description = f'OpenEvals {OPENEVALS_VERSION} adherence of the executed steps to the expected plan.'

  def __init__(self, *, judge: OpenEvalsJudge, output_token_limit: int):
    super().__init__(judge_client=NO_SYLLO_EVAL_JUDGE)
    self._evaluator = create_async_llm_as_judge(
      prompt=PLAN_ADHERENCE_PROMPT, feedback_key='plan_adherence', judge=judge
    )
    self._output_token_limit = output_token_limit

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    # The steps and the plan render as Plan Correctness renders them.
    call = judged(
      self._evaluator,
      inputs=display_text(span.semantics.request),
      outputs=extract_actual_plan(span),
      plan=ground_truth_text(ground_truths.get(self.ground_truth_keys[0]), 'expected_plan') or '<empty>',
    )
    return await verdict_result(call, output_token_limit=self._output_token_limit)


def openevals_metrics(
  benchmark: Benchmark, judge: OpenEvalsJudge, *, output_token_limit: int
) -> dict[str, EvaluationMetric]:
  """The baseline's metrics for ``benchmark`` by key: one for each judge metric of the main arm on that benchmark."""
  if benchmark == 'tau2':
    return {'plan': OpenEvalsPlanAdherence(judge=judge, output_token_limit=output_token_limit)}
  recall = OpenEvalsGoldClaimsGroundedness if benchmark == 'erb' else OpenEvalsOrbitalsClaimsGroundedness
  return {
    'relevance': OpenEvalsRetrievalRelevance(
      judge=judge, output_token_limit=output_token_limit, target_span_types=(SEARCH_SPAN_TYPE,)
    ),
    'recall': recall(judge=judge, output_token_limit=output_token_limit),
    'answer': OpenEvalsCorrectness(judge=judge, output_token_limit=output_token_limit),
  }


@asynccontextmanager
async def open_openevals_judge(config: JudgeConfig, settings: GeminiJudgeSettings) -> AsyncIterator[OpenEvalsJudge]:
  """Yield the shared judge as OpenEvals calls it: the model, temperature, concurrency and retries of ``config``."""
  chat_model = build_chat_model(config, settings)
  try:
    yield OpenEvalsJudge(
      chat_model,
      temperature=config.temperature,
      max_concurrent_requests=config.max_concurrent_requests,
      max_attempts=config.max_retries,
    )
  finally:
    await chat_model.async_client.aclose()
