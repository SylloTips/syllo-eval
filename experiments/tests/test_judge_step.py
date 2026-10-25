import io
import os
import tempfile
import unittest
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from syllo_eval.model import EvaluationSampleStatus, EvaluationStatus, MetricComputationStatus

import cli
from cli import main, unresolved_samples
from config import JudgeSamplesConfig, load_config
from judge import judge_metrics
from manifest import Manifest, StepStatus


class JudgeMetricsTest(unittest.TestCase):
  def test_each_arm_scores_the_metrics_of_its_benchmark(self) -> None:
    expected = {
      ('erb', 'main'): [
        'answer_correctness_judge',
        'contextual_precision_document_judge',
        'contextual_recall_gold_claims',
      ],
      ('wixqa', 'main'): [
        'answer_correctness_judge',
        'contextual_precision_document_judge',
        'contextual_recall_orbitals_claims',
      ],
      ('tau2', 'main'): ['plan_correctness_judge'],
      ('erb', 'sc'): ['contextual_precision_document_judge_sc', 'contextual_recall_gold_claims_sc'],
      ('wixqa', 'sc'): ['contextual_precision_document_judge_sc'],
      ('tau2', 'sc'): [],
      ('erb', 'wt'): ['answer_correctness_judge_wt'],
      ('wixqa', 'wt'): [],
      ('tau2', 'wt'): ['plan_correctness_judge_wt'],
    }
    for (benchmark, arm), names in expected.items():
      with self.subTest(benchmark=benchmark, arm=arm):
        metrics = judge_metrics(benchmark, arm, judge=MagicMock(), db_manager=MagicMock(), output_token_limit=100)  # type: ignore[arg-type]
        self.assertEqual([metric.name for metric in metrics], names)


class UnresolvedSamplesTest(unittest.TestCase):
  def test_a_sample_is_resolved_only_when_it_completed_and_every_metric_has_an_outcome(self) -> None:
    completed, failed = EvaluationSampleStatus.COMPLETED, EvaluationSampleStatus.FAILED
    done, skipped, error = (
      MetricComputationStatus.COMPLETED,
      MetricComputationStatus.SKIPPED,
      MetricComputationStatus.FAILED,
    )
    cases: list[tuple[EvaluationSampleStatus, list[tuple[str, MetricComputationStatus, str | None]]]] = [
      (completed, [('a', done, None), ('b', skipped, None)]),
      (completed, [('a', error, 'misaligned'), ('b', done, None)]),
      (completed, [('a', done, None), ('b', error, 'provider')]),
      (completed, [('a', done, None)]),
      (completed, [('a', error, None), ('b', done, None)]),
      # Interrupted after persisting part of its results.
      (failed, [('a', done, None), ('b', done, None)]),
    ]
    samples = [uuid4() for _ in range(len(cases) + 1)]
    run_samples = [
      SimpleNamespace(id=uuid4(), sample_id=sample_id, status=status) for sample_id, (status, _) in zip(samples, cases)
    ]
    computations = [
      SimpleNamespace(evaluation_run_sample_id=run_sample.id, metric=metric, status=status, metadata={'failure': kind})
      for run_sample, (_, outcomes) in zip(run_samples, cases)
      for metric, status, kind in outcomes
    ]

    unresolved = unresolved_samples(samples, run_samples, computations, ['a', 'b'])  # type: ignore[arg-type]

    # The last sample was planned but never started.
    self.assertEqual(unresolved, samples[2:])


class ScopedSamplesTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_selected_sample_without_a_trace_is_left_out(self) -> None:
    traced, untraced = uuid4(), uuid4()
    source = cli._CollectionRun(load_config().configuration('erb/react/gpt'), uuid4(), frozenset({traced}), 1)
    config = load_config().model_copy(update={'judge_samples': JudgeSamplesConfig(pilot_samples={'erb': ('q1', 'q2')})})
    uow = MagicMock()
    uow.__aenter__.return_value = uow
    uow.samples.list_by_dataset = AsyncMock(
      return_value=[SimpleNamespace(input_prompt='p1', id=traced), SimpleNamespace(input_prompt='p2', id=untraced)]
    )
    with (
      patch('cli.UnitOfWork', return_value=uow),
      patch('cli.benchmark_steps.sample_prompts', return_value=['p1', 'p2']),
    ):
      scoped = await cli._scoped_samples(MagicMock(), config, source, Namespace(arm='main', pilot=True, data_dir=None))

    self.assertEqual(scoped, [traced])
    uow.samples.list_by_dataset.assert_awaited_once_with(source.dataset_id)


class JudgeCliTest(unittest.TestCase):
  """Runs `judge` with the database, judge and service replaced by fakes; each repeat makes a new run."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.manifest = Path(self._directory.name) / 'manifest.jsonl'
    self.source_run = uuid4()
    self.runs: list[UUID] = []
    self.service = MagicMock()
    self.service.__aenter__.return_value = self.service
    self.service.repeat_evaluation = AsyncMock(side_effect=self._repeat)
    self.db_manager = MagicMock()
    self.db_manager.health_check = AsyncMock(return_value=True)
    configuration = load_config().configuration('wixqa/react/gpt')
    self.source = cli._CollectionRun(configuration, uuid4(), frozenset({uuid4(), uuid4()}), 2)

  def tearDown(self) -> None:
    self._directory.cleanup()

  async def _repeat(self, **kwargs: Any) -> SimpleNamespace:
    run = SimpleNamespace(id=uuid4(), status=EvaluationStatus.COMPLETED)
    self.runs.append(run.id)
    return SimpleNamespace(run=run, execute=AsyncMock(return_value=run))

  def _judge(self, *args: str, unresolved: list[Any]) -> tuple[int, str]:
    @asynccontextmanager
    async def fake_database(settings: Any) -> AsyncIterator[object]:
      yield self.db_manager

    @asynccontextmanager
    async def fake_judge(config: Any, settings: Any) -> AsyncIterator[object]:
      yield MagicMock()

    output = io.StringIO()
    with (
      patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}),
      patch('cli.load_environment'),
      patch('cli.open_database', fake_database),
      patch('cli.open_judge_client', fake_judge),
      patch('cli.build_service', return_value=self.service),
      patch('cli._collection_run', AsyncMock(return_value=self.source)),
      patch('cli._failed_units', AsyncMock(return_value={})),
      patch('cli._unresolved_samples', AsyncMock(return_value=unresolved)),
      patch('cli._retry_samples', AsyncMock(return_value=unresolved or [uuid4()])),
      patch('cli._scoped_samples', AsyncMock(return_value=None)),
      redirect_stdout(output),
    ):
      argv = ['judge', '--source-run', str(self.source_run), '--manifest', str(self.manifest), *args]
      return main(argv), output.getvalue()

  def test_scores_every_traced_sample_and_records_the_step(self) -> None:
    status, output = self._judge('--arm', 'main', '--repetition', '2', unresolved=[])

    self.assertEqual(status, 0, output)
    self.service.repeat_evaluation.assert_awaited_once_with(
      source_run_id=self.source_run,
      metrics=['answer_correctness_judge', 'contextual_precision_document_judge', 'contextual_recall_orbitals_claims'],
      max_concurrent_samples=None,
      sample_ids=None,
    )
    records = Manifest(self.manifest).records()
    self.assertEqual([record.step for record in records], [f'judge:{self.source_run}:main:2'] * 2)
    self.assertEqual([record.status for record in records], [StepStatus.STARTED, StepStatus.COMPLETED])
    self.assertEqual(records[-1].details['collection_failures'], 2)
    self.assertEqual(records[-1].details['samples'], 2)

  def test_the_step_completes_with_every_attempt_once_retries_resolve_it(self) -> None:
    base = f'judge:{self.source_run}:sc:1'
    self.assertEqual(self._judge('--arm', 'sc', unresolved=[uuid4()])[0], 1)
    self.assertEqual(self._judge('--arm', 'sc', '--retry', str(self.runs[0]), unresolved=[uuid4()])[0], 1)
    status, output = self._judge('--arm', 'sc', '--retry', str(self.runs[1]), unresolved=[])

    self.assertEqual(status, 0, output)
    latest = Manifest(self.manifest).latest()
    self.assertEqual(latest[f'{base}:retry-2'].status, StepStatus.COMPLETED)
    self.assertEqual(latest[base].status, StepStatus.COMPLETED)
    self.assertEqual(list(latest[base].run_ids), self.runs)
    self.assertEqual(latest[base].details['metrics'], ['contextual_precision_document_judge_sc'])
    self.assertEqual(latest[base].details['unresolved'], [])

  def test_a_retry_must_name_the_latest_attempt_of_its_own_step(self) -> None:
    self._judge('--arm', 'sc', '--pilot', unresolved=[uuid4()])
    self._judge('--arm', 'main', unresolved=[uuid4()])
    self._judge('--arm', 'main', '--retry', str(self.runs[1]), unresolved=[uuid4()])
    pilot_run, original_run, _ = self.runs

    for args in (
      ('--arm', 'sc', '--retry', str(pilot_run)),
      ('--arm', 'main', '--repetition', '2', '--retry', str(original_run)),
      ('--arm', 'main', '--retry', str(original_run)),
    ):
      with self.subTest(args=args):
        self.assertEqual(self._judge(*args, unresolved=[])[0], 2)
    self.assertEqual(len(self.runs), 3)
    latest = Manifest(self.manifest).latest()
    self.assertNotIn(f'judge:{self.source_run}:sc:1', latest)
    self.assertEqual(latest[f'judge:{self.source_run}:main:1'].status, StepStatus.FAILED)

  def test_refuses_repetitions_outside_the_main_arm(self) -> None:
    status, _ = self._judge('--arm', 'wt', '--repetition', '2', unresolved=[])
    self.assertEqual(status, 2)
    self.service.repeat_evaluation.assert_not_awaited()
