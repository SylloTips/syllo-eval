import unittest
from datetime import datetime, timezone
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
    argv = ['--agent-name', 'demo-agent', '--agent-version-tag', 'v1', '--dataset-id', str(uuid4())]
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

  def test_excluded_root_span_names_are_normalized(self):
    env = {'PHOENIX_REQUEST_ID_EXCLUDED_ROOT_SPAN_NAMES': '[" Helper_Root "]'}
    with patch.dict('os.environ', env, clear=True):
      settings = Settings()
    self.assertEqual(settings.phoenix.request_id_excluded_root_span_names, ('helper_root',))
