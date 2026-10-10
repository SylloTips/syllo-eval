import io
import json
import tempfile
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from syllo_eval.evaluation.trace_import import load_trace_file
from syllo_eval.model import EvaluationStatus
from syllo_eval.settings import PhoenixSettings

import collect
from cli import main
from config import load_config
from manifest import Manifest, StepStatus
from run_outputs import RunOutputs


class FakeTraceClient:
  def __init__(self, records: list[dict[str, Any]]):
    self.records = records

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    return f'trace-of-{request_id}'

  async def get_trace_json(self, trace_id: str) -> list[dict[str, Any]]:
    return self.records


class CollectTest(unittest.IsolatedAsyncioTestCase):
  def test_each_benchmark_has_its_own_project_and_lookups_reach_a_day_back(self) -> None:
    phoenix = collect.phoenix_settings(PhoenixSettings(project_id='default'), 'tau2-retail-v1.0.1', 'tau2-llm-agent')
    longer = collect.phoenix_settings(
      PhoenixSettings(request_id_lookup_time_window_seconds=200_000), 'erb', 'tau2-llm-agent'
    )

    self.assertEqual((phoenix.project_id, phoenix.request_id_lookup_time_window_seconds), ('tau2-retail-v1.0.1', 86400))
    self.assertEqual(longer.request_id_lookup_time_window_seconds, 200_000)

  def test_each_stack_looks_traces_up_by_its_own_request_id_attribute(self) -> None:
    base = PhoenixSettings(request_id_attribute='set.elsewhere')

    self.assertEqual(collect.phoenix_settings(base, 'erb', 'react').request_id_attribute, 'dify_trace_id')
    self.assertEqual(collect.phoenix_settings(base, 'tau2', 'tau2-llm-agent').request_id_attribute, 'request_id')

  def test_collection_computes_only_the_metrics_that_need_no_judge(self) -> None:
    tau2 = collect.deterministic_metrics('tau2')
    search = collect.deterministic_metrics('erb')

    self.assertEqual([metric.name for metric in tau2], ['plan_efficiency'])
    self.assertEqual(
      [metric.name for metric in search], ['set_precision_document', 'set_recall_document', 'ndcg_at_10_document']
    )
    self.assertEqual({metric.target_span_types for metric in search}, {('retrieval',)})

  def test_stacks_without_a_caller_are_reported(self) -> None:
    config = load_config()
    unwired = config.configuration('erb/react/sonnet').model_copy(update={'agent': 'unwired'})

    with self.assertRaisesRegex(collect.UnsupportedAgentError, 'the unwired agent stack has no caller yet'):
      collect.build_integration(
        unwired,
        config,
        data_dir=Path('data'),
        phoenix=PhoenixSettings(),
        outputs=RunOutputs(Path('outputs')),
      )

  async def test_the_archive_keeps_each_fetched_trace_in_the_import_format(self) -> None:
    record = {'context.span_id': 's1', 'name': 'root', 'start_time': datetime(2026, 10, 1, tzinfo=timezone.utc)}
    with tempfile.TemporaryDirectory() as directory:
      outputs = RunOutputs(Path(directory))
      archive = collect.TraceArchive(FakeTraceClient([record]), outputs)
      with self.assertRaisesRegex(RuntimeError, 'not been created'):
        await archive.get_trace_json('t1')
      run_id = uuid4()
      outputs.bind(run_id)

      trace_id = await archive.get_trace_id_by_request_id('r1')
      records = await archive.get_trace_json(str(trace_id))
      archived = load_trace_file(Path(directory) / 'traces' / str(run_id) / 'trace-of-r1.json')

    self.assertEqual(records, [record])
    self.assertEqual(archived.trace_id, 'trace-of-r1')
    self.assertEqual(archived.spans, [{**record, 'start_time': '2026-10-01T00:00:00Z'}])


class CollectCliTest(unittest.TestCase):
  """Runs `collect` with the database, service and agent replaced by fakes."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.root = Path(self._directory.name)
    self.manifest = self.root / 'manifest.jsonl'
    self.run_id, self.dataset_id = uuid4(), uuid4()
    self.service = MagicMock()
    self.service.__aenter__.return_value = self.service
    self.service.create_evaluation = AsyncMock()
    self.build_service = MagicMock(return_value=self.service)
    self.db_manager = MagicMock()
    self.db_manager.health_check = AsyncMock(return_value=True)
    self.dataset: SimpleNamespace | None = SimpleNamespace(id=self.dataset_id)
    self.integration = collect.AgentIntegration(caller=MagicMock(), adapter=MagicMock())
    self.integration_error: Exception | None = None

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _finishing(self, status: EvaluationStatus, completed: int, failed: int) -> None:
    run = SimpleNamespace(id=self.run_id, status=status)
    self.service.create_evaluation.return_value = SimpleNamespace(run=run, execute=AsyncMock(return_value=run))
    counts = SimpleNamespace(total=completed + failed, completed=completed, failed=failed)
    self.service.get_evaluation_status = AsyncMock(return_value=SimpleNamespace(sample_counts=counts))

  def _collect(self, *args: str) -> tuple[int, str, str]:
    @asynccontextmanager
    async def fake_database(settings: Any) -> AsyncIterator[object]:
      yield self.db_manager

    uow = MagicMock()
    uow.__aenter__.return_value = uow
    uow.datasets.get_by_name = AsyncMock(return_value=self.dataset)
    self.uow = uow
    output, errors = io.StringIO(), io.StringIO()
    with (
      patch('cli.load_environment'),
      patch('cli.logging.basicConfig'),
      patch('cli.open_database', fake_database),
      patch('cli.build_service', self.build_service),
      patch('cli.UnitOfWork', return_value=uow),
      patch(
        'cli.collect.build_integration', return_value=self.integration, side_effect=self.integration_error
      ) as build_integration,
      redirect_stdout(output),
      redirect_stderr(errors),
    ):
      status = main(['collect', '--manifest', str(self.manifest), '--outputs-dir', str(self.root / 'outputs'), *args])
    self.build_integration = build_integration
    return status, output.getvalue(), errors.getvalue()

  def test_runs_the_configuration_on_its_dataset_and_records_the_run(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED, completed=114, failed=0)

    status, output, _ = self._collect(
      '--configuration', 'tau2/llm-agent/sonnet/trial-1', '--max-concurrent-samples', '4'
    )

    self.assertEqual(status, 0, output)
    phoenix = self.build_integration.call_args.kwargs['phoenix']
    self.assertEqual(phoenix.project_id, 'tau2-retail-v1.0.1')
    kwargs = self.build_service.call_args.kwargs
    self.assertEqual([metric.name for metric in kwargs['metrics']], ['plan_efficiency'])
    self.assertEqual(kwargs['callers_by_agent_name'], {'tau2-llm-agent': self.integration.caller})
    self.assertEqual(kwargs['trace_adapters_by_agent_name'], {'tau2-llm-agent': self.integration.adapter})
    self.assertIsInstance(kwargs['trace_client'], collect.TraceArchive)
    self.service.create_evaluation.assert_awaited_once_with(
      agent_name='tau2-llm-agent',
      agent_version_tag='sonnet-trial1',
      dataset_id=self.dataset_id,
      max_concurrent_samples=4,
      selected_metric_names=['plan_efficiency'],
    )
    records = Manifest(self.manifest).records()
    self.assertEqual([record.status for record in records], [StepStatus.STARTED, StepStatus.COMPLETED])
    self.assertEqual({record.step for record in records}, {'collect:tau2/llm-agent/sonnet/trial-1'})
    self.assertEqual(records[-1].run_ids, (self.run_id,))
    self.assertEqual(records[-1].details['samples'], {'total': 114, 'completed': 114, 'failed': 0})
    self.assertIn(f'run {self.run_id} COMPLETED: 114 of 114 samples completed', output)

  def test_a_pilot_runs_on_the_pilot_dataset_and_is_recorded_apart(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED, completed=3, failed=0)

    status, output, _ = self._collect('--configuration', 'tau2/llm-agent/sonnet/trial-1', '--pilot')

    self.assertEqual(status, 0, output)
    self.uow.datasets.get_by_name.assert_awaited_once_with('tau2-retail-v1.0.1-pilot')
    # Agents keep sending spans to the benchmark's project.
    self.assertEqual(self.build_integration.call_args.kwargs['phoenix'].project_id, 'tau2-retail-v1.0.1')
    record = Manifest(self.manifest).records()[-1]
    self.assertEqual(
      (record.step, record.status), ('collect:tau2/llm-agent/sonnet/trial-1:pilot', StepStatus.COMPLETED)
    )
    self.assertEqual(record.details['dataset_name'], 'tau2-retail-v1.0.1-pilot')

  def test_failed_samples_leave_the_step_failed(self) -> None:
    self._finishing(EvaluationStatus.PARTIALLY_COMPLETED, completed=112, failed=2)

    status, output, _ = self._collect('--configuration', 'tau2/llm-agent/sonnet/trial-1')

    self.assertEqual(status, 1)
    self.assertEqual(Manifest(self.manifest).records()[-1].status, StepStatus.FAILED)
    self.assertIn('the failed ones must run again', output)

  def test_a_dataset_that_is_not_imported_fails_before_any_run(self) -> None:
    self.dataset = None

    status, output, _ = self._collect('--configuration', 'tau2/llm-agent/sonnet/trial-1')

    self.assertEqual(status, 1)
    self.service.create_evaluation.assert_not_awaited()
    [record] = Manifest(self.manifest).records()
    self.assertEqual(record.status, StepStatus.FAILED)
    self.assertIn('benchmarks import --only tau2', json.dumps(record.details))

  def test_an_unknown_configuration_is_a_usage_error(self) -> None:
    status, _, errors = self._collect('--configuration', 'tau2/missing')

    self.assertEqual(status, 2)
    self.assertIn("Unknown configuration 'tau2/missing'", errors)

  def test_a_stack_without_a_caller_is_a_usage_error(self) -> None:
    self.integration_error = collect.UnsupportedAgentError('erb/react/sonnet: the react agent stack has no caller yet')

    status, _, errors = self._collect('--configuration', 'erb/react/sonnet')

    self.assertEqual(status, 2)
    self.assertIn('collect: erb/react/sonnet: the react agent stack has no caller yet', errors)
    self.assertFalse(self.manifest.exists())


if __name__ == '__main__':
  unittest.main()
