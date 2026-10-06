from collections.abc import Mapping, Sequence

from syllo_eval.evaluation.claim_extractor import (
  ClaimExtractionResult,
  ClaimExtractorClient,
  ClaimExtractorMessage,
  ExtractedClaim,
)
from syllo_eval.evaluation.judge import LlmJudgeClient
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.evaluation.metric_support.retrieved_context import retrieval_skip_result
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import BaseContextualRecallJudgeMetric
from syllo_eval.model import GroundTruth, Span


class _BaseContextualRecallClaimExtractorMetric(BaseContextualRecallJudgeMetric):
  """Scores Orbitals-extracted expected-answer claims supported by retrieved context."""

  requires_claim_extractor_client = True

  def __init__(
    self,
    *,
    judge_client: LlmJudgeClient,
    claim_extractor_client: ClaimExtractorClient,
    rubric_addition: str | None = None,
    target_span_types: Sequence[str] = ('agent_root',),
  ):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition, target_span_types=target_span_types)
    self._claim_extractor_client = claim_extractor_client

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    ground_truth = ground_truths[self.ground_truth_keys[0]]

    skip_result = retrieval_skip_result(
      span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage, require_content=True
    )
    if skip_result is not None:
      return skip_result
    retrieved_items = self._extract_retrieved_items(span)
    if not retrieved_items:
      # Nothing retrieved can support a claim, so recall is 0 without calling Orbitals or the judge.
      return self._build_result(retrieved_items, [], [], [])

    extraction = await self._claim_extractor_client.extract(
      [
        ClaimExtractorMessage(role='user', content=span.semantics.request or ''),
        ClaimExtractorMessage(role='assistant', content=ground_truth.ground_truth_value['expected_output']),
      ]
    )
    claims = [claim for claim in extraction.claims if claim.subtype.lower() != 'unverifiable']
    excluded_claim_count = len(extraction.claims) - len(claims)
    if not claims:
      return self._enrich_result(
        self._build_result(retrieved_items, [], [], []), extraction, claims, excluded_claim_count
      )

    judged = await self.judge_claims(span, retrieved_items, [claim.content for claim in claims])
    if isinstance(judged, MetricComputationResult):
      return self._enrich_result(judged, extraction, claims, excluded_claim_count, rewrite_reason=False)
    judge_results, judgments = judged

    return self._enrich_result(
      self._build_result(retrieved_items, [claim.content for claim in claims], judgments, judge_results),
      extraction,
      claims,
      excluded_claim_count,
    )

  @staticmethod
  def _enrich_result(
    result: MetricComputationResult,
    extraction: ClaimExtractionResult,
    claims: list[ExtractedClaim],
    excluded_claim_count: int,
    *,
    rewrite_reason: bool = True,
  ) -> MetricComputationResult:
    metadata = dict(result.metadata or {})
    counts = dict(metadata.get('counts', {}))
    counts['verifiable_claims'] = counts.pop('expected_statements', len(claims))
    counts['excluded_unverifiable_claims'] = excluded_claim_count
    metadata['counts'] = counts
    metadata['claim_extractor'] = {
      'provider': 'orbitals',
      'model': extraction.model,
      'usage': extraction.usage,
      'time_taken': extraction.time_taken,
    }
    for claim_result, claim in zip(metadata.get('statement_results', []), claims):
      claim_result['subtype'] = claim.subtype
    result.metadata = metadata
    result.raw_output = {**(result.raw_output or {}), 'claim_extraction': extraction.model_dump()}
    if rewrite_reason:
      attributable_count = counts.get('attributable', 0)
      result.reasoning = (
        f'Contextual recall over {len(claims)} verifiable Orbitals claims '
        f'with {attributable_count} attributable claims.'
      )
    return result


class ContextualRecallDocumentClaimExtractorMetric(_BaseContextualRecallClaimExtractorMetric):
  variant = 'document'
  metric_name = 'contextual_recall_document_claim_extractor'
  metric_description = 'Uses Orbitals ClaimExtractor and an LLM judge to compute contextual recall over documents.'


class ContextualRecallSnippetClaimExtractorMetric(_BaseContextualRecallClaimExtractorMetric):
  variant = 'snippet'
  metric_name = 'contextual_recall_snippet_claim_extractor'
  metric_description = 'Uses Orbitals ClaimExtractor and an LLM judge to compute contextual recall over snippets.'
