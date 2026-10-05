import unittest

from syllo_eval.evaluation.claim_extractor import ClaimExtractionResult, ClaimExtractorMessage
from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.metrics.available import (
  BUILTIN_METRICS,
  available_metric_names,
  build_available_metrics,
  required_clients,
)


class _FakeJudgeClient:
  async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
    raise AssertionError('test only instantiates judge metrics')

  async def aclose(self) -> None:
    pass


class _FakeClaimExtractorClient:
  async def extract(self, conversation: list[ClaimExtractorMessage]) -> ClaimExtractionResult:
    del conversation
    raise AssertionError('test only instantiates claim extractor metrics')

  async def aclose(self) -> None:
    pass


class AvailableMetricsTest(unittest.TestCase):
  def test_available_metric_names_include_judge_metrics_only_when_enabled(self) -> None:
    deterministic_names = available_metric_names(llm_judge_enabled=False)
    all_names = available_metric_names(llm_judge_enabled=True)

    self.assertIn('plan_efficiency', deterministic_names)
    self.assertNotIn('answer_correctness_judge', deterministic_names)
    self.assertIn('answer_correctness_judge', all_names)
    self.assertNotIn('contextual_recall_document_claim_extractor', all_names)

  def test_available_metric_names_include_claim_extractor_metrics_when_enabled(self) -> None:
    metric_names = available_metric_names(llm_judge_enabled=True, claim_extractor_enabled=True)

    self.assertIn('contextual_recall_document_claim_extractor', metric_names)
    self.assertIn('contextual_recall_snippet_claim_extractor', metric_names)

  def test_build_available_metrics_instantiates_judge_metrics_with_client(self) -> None:
    metrics = build_available_metrics(
      judge_client=_FakeJudgeClient(),
      claim_extractor_client=_FakeClaimExtractorClient(),
    )
    metric_names = {metric.name for metric in metrics}

    self.assertIn('plan_efficiency', metric_names)
    self.assertIn('answer_correctness_judge', metric_names)
    self.assertIn('contextual_recall_document_claim_extractor', metric_names)


class MetricClientRequirementsTest(unittest.TestCase):
  def test_deterministic_selection_requires_no_clients(self) -> None:
    requirements = required_clients(['plan_efficiency', 'set_recall_document'])

    self.assertFalse(requirements.judge)
    self.assertFalse(requirements.claim_extractor)

  def test_judge_metric_requires_only_the_judge_client(self) -> None:
    requirements = required_clients(['plan_efficiency', 'answer_correctness_judge'])

    self.assertTrue(requirements.judge)
    self.assertFalse(requirements.claim_extractor)

  def test_claim_extractor_metric_also_requires_the_judge_client(self) -> None:
    """Inherited from the judge base class rather than restated as a separate rule."""
    requirements = required_clients(['contextual_recall_document_claim_extractor'])

    self.assertTrue(requirements.judge)
    self.assertTrue(requirements.claim_extractor)

  def test_unknown_and_empty_selections_require_no_clients(self) -> None:
    self.assertFalse(required_clients([]).judge)
    self.assertFalse(required_clients(['not_a_metric']).judge)

  def test_metric_names_are_normalized_before_matching(self) -> None:
    self.assertTrue(required_clients(['  ANSWER_CORRECTNESS_JUDGE  ']).judge)


class BuiltinMetricDeclarationTest(unittest.TestCase):
  def test_every_builtin_declares_a_unique_non_empty_name(self) -> None:
    """``name`` now defaults to the class-level ``metric_name``; an unset one would silently be ''."""
    names = [metric_class.metric_name for metric_class in BUILTIN_METRICS]

    self.assertTrue(all(names), f'built-in metric missing metric_name: {names}')
    self.assertEqual(len(names), len(set(names)))

  def test_declared_dependencies_match_what_construction_needs(self) -> None:
    """A metric that needs a client but forgets to declare it would fail here, not at compute time."""
    for metric_class in BUILTIN_METRICS:
      with self.subTest(metric=metric_class.metric_name):
        dependencies: dict[str, object] = {}
        if metric_class.requires_judge_client:
          dependencies['judge_client'] = _FakeJudgeClient()
        if metric_class.requires_claim_extractor_client:
          dependencies['claim_extractor_client'] = _FakeClaimExtractorClient()

        metric = metric_class(**dependencies)  # type: ignore[arg-type]

        self.assertEqual(metric.name, metric_class.metric_name)


class RetrievalSpanTypesTest(unittest.TestCase):
  def test_retrieval_metrics_score_the_configured_span_types(self) -> None:
    metrics = build_available_metrics(
      judge_client=_FakeJudgeClient(),
      claim_extractor_client=_FakeClaimExtractorClient(),
      retrieval_span_types=('retrieval',),
    )

    targets = {metric.name: metric.target_span_types for metric in metrics}
    retrieval = {metric.name for metric in metrics if type(metric).accepts_target_span_types}
    self.assertEqual(len(retrieval), 14)
    self.assertEqual({targets[name] for name in retrieval}, {('retrieval',)})
    self.assertEqual(targets['answer_correctness_judge'], ('agent_root',))
