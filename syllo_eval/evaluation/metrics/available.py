from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from syllo_eval.evaluation.claim_extractor import ClaimExtractorClient
from syllo_eval.evaluation.judge import LlmJudgeClient
from syllo_eval.evaluation.metrics.contracts import EvaluationMetric
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.demo.llm_calls import LlmCallsMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.efficiency import PlanEfficiencyMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionSnippetJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_claim_extractor import (
  ContextualRecallDocumentClaimExtractorMetric,
  ContextualRecallSnippetClaimExtractorMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
  ContextualRecallSnippetJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_stored_claims import (
  ContextualRecallDocumentStoredClaimsMetric,
  ContextualRecallSnippetStoredClaimsMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.ndcg_at_10 import (
  NdcgAt10DocumentMetric,
  NdcgAt10SnippetMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.set_precision import (
  SetPrecisionDocumentMetric,
  SetPrecisionSnippetMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.set_recall import (
  SetRecallDocumentMetric,
  SetRecallSnippetMetric,
)

BUILTIN_METRICS: tuple[type[EvaluationMetric], ...] = (
  LlmCallsMetric,
  PlanEfficiencyMetric,
  SetPrecisionDocumentMetric,
  SetPrecisionSnippetMetric,
  SetRecallDocumentMetric,
  SetRecallSnippetMetric,
  NdcgAt10DocumentMetric,
  NdcgAt10SnippetMetric,
  AnswerCorrectnessJudgeMetric,
  PlanCorrectnessJudgeMetric,
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionSnippetJudgeMetric,
  ContextualRecallDocumentJudgeMetric,
  ContextualRecallSnippetJudgeMetric,
  ContextualRecallDocumentClaimExtractorMetric,
  ContextualRecallSnippetClaimExtractorMetric,
  ContextualRecallDocumentStoredClaimsMetric,
  ContextualRecallSnippetStoredClaimsMetric,
)


@dataclass(frozen=True, slots=True)
class MetricClientRequirements:
  """Which provider clients a set of metrics needs."""

  judge: bool = False
  claim_extractor: bool = False


def _is_available(
  metric_class: type[EvaluationMetric],
  llm_judge_enabled: bool,
  claim_extractor_enabled: bool,
) -> bool:
  if metric_class.requires_judge_client and not llm_judge_enabled:
    return False
  if metric_class.requires_claim_extractor_client and not claim_extractor_enabled:
    return False
  return True


def available_metric_names(llm_judge_enabled: bool, claim_extractor_enabled: bool = False) -> list[str]:
  return [
    metric_class.metric_name
    for metric_class in BUILTIN_METRICS
    if _is_available(metric_class, llm_judge_enabled, claim_extractor_enabled)
  ]


def required_clients(selected_metric_names: Sequence[str]) -> MetricClientRequirements:
  """Report which provider clients the selected built-in metrics need.

  A claim-extractor metric also needs a judge, which its class declares by inheriting
  ``requires_judge_client`` — there is no separate rule here to keep in step.
  """
  selected = {name.strip().lower() for name in selected_metric_names}
  chosen = [metric_class for metric_class in BUILTIN_METRICS if metric_class.metric_name.lower() in selected]
  return MetricClientRequirements(
    judge=any(metric_class.requires_judge_client for metric_class in chosen),
    claim_extractor=any(metric_class.requires_claim_extractor_client for metric_class in chosen),
  )


def build_available_metrics(
  judge_client: LlmJudgeClient | None = None,
  claim_extractor_client: ClaimExtractorClient | None = None,
  rubric_additions: Mapping[str, str] | None = None,
  retrieval_span_types: Sequence[str] = ('agent_root',),
) -> list[EvaluationMetric]:
  metrics: list[EvaluationMetric] = []
  for metric_class in BUILTIN_METRICS:
    if not _is_available(metric_class, judge_client is not None, claim_extractor_client is not None):
      continue
    dependencies: dict[str, Any] = {}
    if metric_class.requires_judge_client:
      dependencies['judge_client'] = judge_client
      dependencies['rubric_addition'] = (rubric_additions or {}).get(metric_class.metric_name)
    if metric_class.requires_claim_extractor_client:
      dependencies['claim_extractor_client'] = claim_extractor_client
    if metric_class.accepts_target_span_types:
      dependencies['target_span_types'] = retrieval_span_types
    metrics.append(metric_class(**dependencies))
  return metrics
