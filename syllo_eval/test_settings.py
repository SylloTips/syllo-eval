import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from syllo_eval.settings import (
  DatabaseSettings,
  EvaluationSettings,
  LlmJudgeSettings,
  Settings,
  load_settings_env,
)


class SettingsTest(unittest.TestCase):
  def test_defaults_apply_when_no_environment_is_set(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      settings = Settings()

    self.assertEqual(settings.evaluation.max_concurrent_samples, 1)
    self.assertEqual(settings.evaluation.max_concurrent_tasks, 10)
    self.assertIsNone(settings.evaluation.sample_trace_timeout)
    self.assertIsNone(settings.evaluation.sample_compute_timeout)
    self.assertIsNone(settings.llm_judge.provider)

  def test_deployment_defaults_are_stable(self) -> None:
    """Deployments set only what differs from these defaults, so changing one changes their behavior."""
    with patch.dict(os.environ, {}, clear=True):
      settings = Settings()

    self.assertEqual(settings.phoenix.timeout, 1000)
    self.assertEqual(settings.phoenix.max_retries, 5)
    self.assertEqual(settings.phoenix.trace_fetch_initial_backoff_seconds, 5.0)
    self.assertEqual(settings.phoenix.trace_fetch_max_backoff_seconds, 30.0)
    self.assertEqual(settings.phoenix.request_id_lookup_max_attempts, 10)
    self.assertEqual(settings.phoenix.request_id_lookup_initial_backoff_seconds, 2.0)
    self.assertEqual(settings.phoenix.request_id_lookup_max_backoff_seconds, 30.0)
    self.assertEqual(settings.phoenix.request_id_lookup_time_window_seconds, 600.0)
    self.assertEqual(settings.llm_judge.gemini.model, 'gemini-3.1-flash-lite')
    self.assertEqual(settings.evaluation.max_concurrent_samples, 1)

  def test_explicit_values_take_precedence_over_the_environment(self) -> None:
    with patch.dict(os.environ, {'EVALUATION_MAX_CONCURRENT_SAMPLES': '3'}, clear=True):
      settings = Settings(evaluation=EvaluationSettings(max_concurrent_samples=9))

    self.assertEqual(settings.evaluation.max_concurrent_samples, 9)

  def test_nested_blocks_read_their_own_environment_prefixes(self) -> None:
    with patch.dict(
      os.environ,
      {
        'DB_HOST': 'db.internal',
        'DB_NAME': 'pg-custom',
        'PHOENIX_BASE_URL': 'http://phoenix:6006',
        'LLM_JUDGE_PROVIDER': 'GEMINI',
        'GOOGLE_API_KEY': 'gemini-key',
        'ORBITALS_API_KEY': 'orbitals-key',
      },
      clear=True,
    ):
      settings = Settings()

    self.assertEqual(settings.database.host, 'db.internal')
    self.assertEqual(settings.database.database, 'pg-custom')
    self.assertEqual(settings.phoenix.base_url, 'http://phoenix:6006')
    self.assertEqual(settings.llm_judge.provider, 'gemini')
    self.assertEqual(settings.llm_judge.gemini.api_key, 'gemini-key')
    self.assertEqual(settings.orbitals.api_key, 'orbitals-key')

  def test_blank_and_padded_values_fall_back_to_defaults(self) -> None:
    with patch.dict(os.environ, {'DB_HOST': '   ', 'DB_NAME': '  pg-padded  '}, clear=True):
      settings = DatabaseSettings()

    self.assertEqual(settings.host, 'localhost')
    self.assertEqual(settings.database, 'pg-padded')

  def test_rejects_unsupported_judge_provider(self) -> None:
    with patch.dict(os.environ, {'LLM_JUDGE_PROVIDER': 'anthropic'}, clear=True):
      with self.assertRaises(ValueError):
        LlmJudgeSettings()

  def test_blank_nested_environment_values_are_unset_before_json_decoding(self) -> None:
    with patch.dict(
      os.environ,
      {
        'DATABASE': '',
        'LLM_JUDGE_OPENAI': '',
        'DB_HOST': 'pod-db',
        'OPENAI_API_KEY': 'synthetic-key',
        'LLM_JUDGE_PROVIDER': '   ',
      },
      clear=True,
    ):
      settings = Settings()
    self.assertEqual(settings.database.host, 'pod-db')
    self.assertEqual(settings.llm_judge.openai.api_key, 'synthetic-key')
    self.assertIsNone(settings.llm_judge.provider)

  def test_blank_dotenv_nested_value_does_not_override_environment_or_defaults(self) -> None:
    with TemporaryDirectory() as directory:
      path = Path(directory) / '.env'
      path.write_text('DATABASE=\n', encoding='utf-8')
      with patch.dict(os.environ, {'DB_HOST': 'pod-db'}, clear=True):
        settings = Settings(_env_file=path)  # type: ignore[call-arg]  # BaseSettings runtime option
    self.assertEqual(settings.database.host, 'pod-db')

  def test_nonblank_nested_json_and_explicit_overrides_are_preserved(self) -> None:
    with patch.dict(os.environ, {'DATABASE': '{"host":"json-db"}', 'DB_PORT': '5433'}, clear=True):
      self.assertEqual(Settings().database.host, 'json-db')
      explicit = Settings(database=DatabaseSettings(host='explicit-db'))
    self.assertEqual(explicit.database.host, 'explicit-db')
    self.assertEqual(explicit.database.port, 5433)

  def test_invalid_nonblank_nested_json_is_still_rejected(self) -> None:
    from pydantic_settings import SettingsError

    with patch.dict(os.environ, {'DATABASE': 'not-json'}, clear=True):
      with self.assertRaises(SettingsError):
        Settings()

  def test_evaluation_settings_read_concurrency_and_timeout_options(self) -> None:
    with patch.dict(
      os.environ,
      {
        'EVALUATION_MAX_CONCURRENT_SAMPLES': '3',
        'EVALUATION_MAX_CONCURRENT_TASKS': '7',
        'EVALUATION_SAMPLE_TRACE_TIMEOUT_SECONDS': '12.5',
        'EVALUATION_SAMPLE_COMPUTE_TIMEOUT_SECONDS': '9',
      },
      clear=True,
    ):
      settings = EvaluationSettings()

    self.assertEqual(settings.max_concurrent_samples, 3)
    self.assertEqual(settings.max_concurrent_tasks, 7)
    self.assertEqual(settings.sample_trace_timeout, 12.5)
    self.assertEqual(settings.sample_compute_timeout, 9.0)

  def test_load_settings_env_accepts_explicit_dotenv_path(self) -> None:
    with TemporaryDirectory() as directory:
      dotenv_path = Path(directory) / '.env'
      dotenv_path.write_text('EVALUATION_MAX_CONCURRENT_TASKS=4\n')
      with patch.dict(os.environ, {}, clear=True):
        load_settings_env(dotenv_path=dotenv_path)
        settings = EvaluationSettings()

    self.assertEqual(settings.max_concurrent_tasks, 4)


if __name__ == '__main__':
  unittest.main()
