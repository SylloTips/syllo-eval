import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter
from syllo_eval.evaluation.trace_processor import TraceProcessor
from syllo_eval.evaluation.metric_planner import MetricPlanner
from syllo_eval.evaluation.metrics.contracts import SpanGroupEvaluationMetric, MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.rag.set_precision import (
  SetPrecisionDocumentMetric,
  SetPrecisionSnippetMetric,
)
from syllo_eval.evaluation.metrics.test_support import span, truth
from syllo_eval.infrastructure.arize import PhoenixClient
from syllo_eval.settings import PhoenixSettings
from syllo_eval.trace_semantics import PlanSnapshot, PlannedStep, RetrievalResult, RetrievalItem


def record(id, parent=None, name='custom', output=None, **fields):
  now = datetime(2026, 9, 18, tzinfo=timezone.utc)
  return {
    'context.span_id': id,
    'parent_id': parent,
    'name': name,
    'start_time': now,
    'end_time': now,
    'attributes.output.value': output,
    **fields,
  }


class TraceNormalizationTest(unittest.TestCase):
  def test_provider_decoder_preserves_structure_and_does_not_guess_agent_semantics(self):
    result = PhoenixTraceAdapter().normalize(
      't',
      [
        record('child', 'root', output={'items': []}),
        record('root', **{'attributes.final_state': {'answer': 'hidden'}}),
      ],
    )
    self.assertEqual([s.external_id for s in result.spans], ['root', 'child'])
    self.assertIsNone(result.spans[0].semantics.answer)
    self.assertEqual(result.spans[1].output_data, {'items': []})
    with self.assertRaises(ValueError):
      PhoenixTraceAdapter().normalize('t', [record('child', 'missing')])

  def test_invalid_canonical_dependencies_are_rejected(self):
    with self.assertRaises(ValueError):
      PlanSnapshot(id='p', steps=[PlannedStep(id='a', operation='search', depends_on=['b'])])
    with self.assertRaises(ValueError):
      PlanSnapshot(id='p', steps=[PlannedStep(id='a', operation='search', depends_on=['a'])])
    with self.assertRaises(ValueError):
      RetrievalResult(kind='document', availability='unavailable', items=[RetrievalItem(id='d')])


class DocumentGroup(SpanGroupEvaluationMetric):
  metric_name = 'group_recall'

  @property
  def target_span_types(self):
    return ('retrieval',)

  @property
  def requires_ground_truth(self):
    return False

  def matches_span(self, span):
    return any(result.kind == 'document' for result in span.semantics.retrieval)

  async def compute(self, spans, ground_truth):
    ids = {item.id for s in spans for result in s.semantics.retrieval for item in result.items}
    return MetricComputationResult(score=float(len(ids)))


class SearchDocumentPrecision(SetPrecisionDocumentMetric):
  retrieval_stage = 'retrieved'

  @property
  def target_span_types(self):
    return ('retrieval',)


class SearchSnippetPrecision(SetPrecisionSnippetMetric):
  retrieval_stage = 'retrieved'

  @property
  def target_span_types(self):
    return ('retrieval',)


class SelectionTest(unittest.IsolatedAsyncioTestCase):
  async def test_document_snippet_and_group_targets_share_the_same_trace(self):
    spans = []
    for id, kind, ids in [('a', 'document', ['d1', 'd2']), ('b', 'snippet', ['s1']), ('c', 'document', ['d2', 'd3'])]:
      target = span(retrieval=[RetrievalResult(kind=kind, items=[RetrievalItem(id=i) for i in ids])])
      target.external_id = id
      spans.append(target)
    metrics = [SearchDocumentPrecision(), SearchSnippetPrecision(), DocumentGroup()]
    truths = [truth(relevant_ids=['d1']), truth(relevant_ids=['s1'])]
    truths[0].key = 'relevant_document_ids'
    truths[1].key = 'relevant_snippet_ids'
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.spans.list_by_trace.return_value = spans
    uow.ground_truths.list_by_sample.return_value = truths
    registry = SimpleNamespace(list_registered=lambda: metrics)
    planner = MetricPlanner(cast(Any, registry), cast(Any, object()))
    with patch('syllo_eval.evaluation.metric_planner.UnitOfWork', return_value=uow):
      items = [item async for item in planner.iter_plan_items('t', uuid4())]
    self.assertEqual([item.target.span_ids for item in items], [['a'], ['c'], ['b'], ['a', 'c']])
    self.assertTrue(all(item.skip_reason is None for item in items))
    self.assertEqual((await metrics[-1].compute(items[-1].target.compute_input, None)).score, 3)

    # The group hook can partition the same selected spans without rewriting ingestion.
    class PartitionedGroup(DocumentGroup):
      def group_spans(self, selected):
        return [(s,) for s in selected]

    partitioned = PartitionedGroup()
    self.assertEqual(len(planner._build_targets(partitioned, 'retrieval', [spans[0], spans[2]])), 2)

  async def test_builtin_rag_targets_final_context_and_custom_metrics_can_score_searches(self):
    raw = span(retrieval=[RetrievalResult(kind='document', items=[RetrievalItem(id='noise')])])
    final = span(
      retrieval=[
        RetrievalResult(kind='document', items=[RetrievalItem(id='noise')]),
        RetrievalResult(kind='document', stage='selected', items=[RetrievalItem(id='right')]),
      ]
    )
    final.span_type = 'agent_root'
    metric = SetPrecisionDocumentMetric()
    planner = MetricPlanner(cast(Any, object()), cast(Any, object()))
    self.assertFalse(metric.matches_span(raw))
    self.assertEqual(metric.target_span_types, ('agent_root',))
    self.assertEqual(len(planner._build_targets(metric, 'agent_root', [final])), 1)
    self.assertEqual((await metric.compute(final, truth(relevant_ids=['right']))).score, 1)
    self.assertEqual((await SearchDocumentPrecision().compute(raw, truth(relevant_ids=['right']))).score, 0)

  async def test_unavailable_target_is_retained_as_a_skip(self):
    target = span(
      retrieval=[RetrievalResult(kind='document', stage='selected', availability='unavailable', reason='not recorded')]
    )
    metric = SetPrecisionDocumentMetric()
    self.assertTrue(metric.matches_span(target))
    self.assertIn('not recorded', metric.input_skip_reason(target) or '')


class IngestionTest(unittest.IsolatedAsyncioTestCase):
  async def test_phoenix_fetch_preserves_status_through_normalization(self):
    client = PhoenixClient(PhoenixSettings())
    for source, expected in [('OK', 'success'), ('ERROR', 'error'), ('UNSET', 'unset'), (None, 'unset')]:
      with self.subTest(status=source):
        node = {
          'name': 'root',
          'spanKind': 'agent',
          'statusCode': source,
          'statusMessage': 'Source detail',
          'context': {'spanId': 's', 'traceId': 't'},
          'startTime': '2026-09-21T10:00:00Z',
          'endTime': '2026-09-21T10:00:01Z',
        }
        response = {
          'data': {
            'getTraceByOtelId': {
              'spans': {
                'edges': [{'node': node}],
                'pageInfo': {'hasNextPage': False},
              }
            }
          }
        }
        with patch.object(client, '_post_graphql', new=AsyncMock(return_value=response)) as post:
          records = await client.get_trace_json('t')
        self.assertIn('statusCode', post.call_args.args[0])
        self.assertIn('statusMessage', post.call_args.args[0])
        normalized = PhoenixTraceAdapter().normalize('t', records).spans[0]
        self.assertEqual(normalized.status, expected)
        self.assertEqual((normalized.metadata or {})['phoenix.status_message'], 'Source detail')

  async def test_identical_ingestion_is_idempotent_and_conflicting_content_is_rejected(self):
    records = [record('root')]
    normalized = PhoenixTraceAdapter().normalize('t', records)
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.traces.get_by_id.return_value = normalized.trace
    uow.spans.list_by_trace.return_value = normalized.spans
    client = AsyncMock()
    client.get_trace_id_by_request_id.return_value = 't'
    client.get_trace_json.return_value = records
    processor = TraceProcessor(client, cast(Any, object()))
    with patch('syllo_eval.evaluation.trace_processor.TransactionalUnitOfWork', return_value=uow):
      self.assertEqual(await processor.process_trace('request'), normalized)
      client.get_trace_json.return_value = [record('root', output='changed')]
      with self.assertRaises(ValueError):
        await processor.process_trace('request')
    uow.traces.create.assert_not_awaited()
    uow.spans.bulk_create.assert_not_awaited()
