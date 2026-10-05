import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from syllo_eval.API.app import create_app
from syllo_eval.API.cli import main
from syllo_eval.model import EvaluationStatus
from syllo_eval.settings import Settings


class ServiceFactoryTest(unittest.TestCase):
  def test_cli_builds_the_runtime_with_the_injected_factory(self):
    run = MagicMock(status=EvaluationStatus.COMPLETED, start_time=datetime.now(timezone.utc), end_time=None)
    factory = MagicMock()
    factory.return_value.available_metric_names.return_value = ['custom_metric']
    argv = ['run', '--agent-name', 'demo-agent', '--agent-version-tag', 'v1', '--dataset-id', str(uuid4())]
    argv += ['--metrics', 'custom_metric']
    with (
      patch.dict('os.environ', {}, clear=True),
      patch('syllo_eval.API.cli.load_settings_env'),
      patch('syllo_eval.API.cli._run', new=AsyncMock(return_value=run)) as run_mock,
    ):
      exit_code = main(argv, service_factory=factory)

    self.assertEqual(exit_code, 0)
    factory.assert_called_once()
    self.assertIsInstance(factory.call_args.args[0], Settings)
    self.assertIs(run_mock.call_args.kwargs['service'], factory.return_value)
    self.assertEqual(run_mock.call_args.kwargs['selected_metric_names'], ['custom_metric'])

  def test_cli_import_loads_trace_files_and_passes_the_adapter(self):
    run = MagicMock(status=EvaluationStatus.COMPLETED, start_time=datetime.now(timezone.utc), end_time=None)
    service = MagicMock(initialize=AsyncMock(), close=AsyncMock())
    service.import_evaluation = AsyncMock(return_value=MagicMock(execute=AsyncMock(return_value=run)))
    with tempfile.TemporaryDirectory() as directory:
      trace_path = Path(directory) / 'trace.json'
      trace_path.write_text('{"trace_id": "trace-1", "spans": [{"name": "root"}]}')
      additions_path = Path(directory) / 'additions.json'
      additions_path.write_text('{"answer_correctness_judge": "Cite a source."}')
      argv = ['import', str(trace_path), '--agent-name', 'demo-agent', '--agent-version-tag', 'v1']
      argv += ['--dataset-id', str(uuid4()), '--trace-adapter', 'custom-source', '--metrics', 'custom_metric']
      argv += ['--rubric-additions', str(additions_path)]
      with patch.dict('os.environ', {}, clear=True), patch('syllo_eval.API.cli.load_settings_env'):
        exit_code = main(argv, service_factory=MagicMock(return_value=service))

    self.assertEqual(exit_code, 0)
    kwargs = cast(Any, service.import_evaluation.await_args).kwargs
    self.assertEqual([trace.trace_id for trace in kwargs['traces']], ['trace-1'])
    self.assertEqual(
      (kwargs['trace_adapter_name'], kwargs['selected_metric_names']), ('custom-source', ['custom_metric'])
    )
    self.assertEqual(kwargs['rubric_additions'], {'answer_correctness_judge': 'Cite a source.'})
    service.close.assert_awaited_once()

  def test_app_builds_the_runtime_with_the_injected_factory(self):
    service = MagicMock(
      initialize=AsyncMock(), close=AsyncMock(), fail_orphaned_running_evaluations=AsyncMock(return_value=0)
    )
    factory = MagicMock(return_value=service)
    with (
      patch.dict('os.environ', {}, clear=True),
      patch('syllo_eval.API.app.load_settings_env'),
      TestClient(create_app(service_factory=factory)),
    ):
      pass

    self.assertIsInstance(factory.call_args.args[0], Settings)
    service.initialize.assert_awaited_once()
    service.close.assert_awaited_once()

  def test_cli_and_app_use_the_given_program_name(self):
    for argv, expected in [(['--help'], 'usage: my-eval '), (['dataset', '--help'], 'usage: my-eval dataset ')]:
      with (
        self.subTest(argv=argv),
        patch('syllo_eval.API.cli.load_settings_env'),
        patch('sys.stdout', new_callable=io.StringIO) as stdout,
        self.assertRaises(SystemExit),
      ):
        main(argv, prog='my-eval')
      self.assertTrue(stdout.getvalue().startswith(expected), stdout.getvalue()[:60])

    self.assertEqual(create_app(service=MagicMock(), title='my-eval').title, 'my-eval')

  def test_excluded_root_span_names_are_normalized(self):
    env = {'PHOENIX_REQUEST_ID_EXCLUDED_ROOT_SPAN_NAMES': '[" Helper_Root "]'}
    with patch.dict('os.environ', env, clear=True):
      settings = Settings()
    self.assertEqual(settings.phoenix.request_id_excluded_root_span_names, ('helper_root',))
