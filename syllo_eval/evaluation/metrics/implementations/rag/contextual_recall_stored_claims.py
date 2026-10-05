from collections.abc import Mapping
from typing import Annotated, ClassVar

from pydantic import Field, TypeAdapter

from syllo_eval.evaluation.metric_support.retrieved_context import retrieval_skip_result
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import BaseContextualRecallJudgeMetric
from syllo_eval.model import ExpectedClaim, GroundTruth, GroundTruthKey, Span

_CLAIMS: TypeAdapter[list[ExpectedClaim]] = TypeAdapter(Annotated[list[ExpectedClaim], Field(min_length=1)])


class BaseContextualRecallStoredClaimsMetric(BaseContextualRecallJudgeMetric):
  """Scores whether claims stored as ground truth are supported by retrieved context.

  The claims are read, never decomposed, so a computation makes one judge call per claim. Subclasses can read another
  ground-truth key, holding claims in the same ``{"claims": [{"id", "text"}, ...]}`` form, by setting ``claims_key``.
  """

  claims_key: ClassVar[str] = GroundTruthKey.EXPECTED_CLAIMS.value

  @property
  def ground_truth_keys(self) -> tuple[str, ...]:
    return (self.claims_key,)

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    ground_truth = ground_truths[self.ground_truth_keys[0]]

    skip_result = retrieval_skip_result(
      span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage, require_content=True
    )
    if skip_result is not None:
      return skip_result
    claims = _CLAIMS.validate_python(ground_truth.ground_truth_value.get('claims'))
    texts = [claim.text for claim in claims]
    claim_ids = [claim.id for claim in claims]
    retrieved_items = self._extract_retrieved_items(span)
    if not retrieved_items:
      # Nothing retrieved can support a claim, so recall is 0 without calling the judge.
      return self._build_result(retrieved_items, texts, [], [], claim_ids)

    judged = await self.judge_claims(span, retrieved_items, texts)
    if isinstance(judged, MetricComputationResult):
      return judged
    judge_results, judgments = judged
    return self._build_result(retrieved_items, texts, judgments, judge_results, claim_ids)


class ContextualRecallDocumentStoredClaimsMetric(BaseContextualRecallStoredClaimsMetric):
  """Scores whether retrieved documents support the claims stored for the sample."""

  variant = 'document'
  metric_name = 'contextual_recall_document_stored_claims'
  metric_description = 'Uses an LLM judge to compute contextual recall over retrieved documents, for stored claims.'


class ContextualRecallSnippetStoredClaimsMetric(BaseContextualRecallStoredClaimsMetric):
  """Scores whether retrieved snippets support the claims stored for the sample."""

  variant = 'snippet'
  metric_name = 'contextual_recall_snippet_stored_claims'
  metric_description = 'Uses an LLM judge to compute contextual recall over retrieved snippets, for stored claims.'
