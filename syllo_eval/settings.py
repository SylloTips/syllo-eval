from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_settings_env(override: bool = False, dotenv_path: str | Path | None = None) -> None:
  """Load settings from an explicit dotenv file, defaulting to ``.env`` in the working directory."""
  load_dotenv(dotenv_path=dotenv_path or Path.cwd() / '.env', override=override)


class EnvSettings(BaseSettings):
  """Base class for the settings blocks.

  Blank values are treated as unset, so a deployment can declare a variable without a value
  (an empty ConfigMap entry, an unfilled ``.env`` line) without overriding the default or,
  worse, enabling a feature gated on "is this key set?". ``env_ignore_empty`` drops them before
  nested JSON decoding; the validator below also strips whitespace-only values of scalar fields.

  Fields are populated by field name as well as by environment variable name, so library consumers
  can build any block in Python without going through the environment.
  """

  model_config = SettingsConfigDict(
    extra='ignore', validate_by_name=True, validate_by_alias=True, env_ignore_empty=True
  )

  @model_validator(mode='before')
  @classmethod
  def _drop_blank_values(cls, data: Any) -> Any:
    if not isinstance(data, dict):
      return data

    cleaned: dict[str, Any] = {}
    for key, value in data.items():
      if not isinstance(value, str):
        cleaned[key] = value
        continue

      stripped = value.strip()
      if stripped:
        cleaned[key] = stripped

    return cleaned


class DatabaseSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='DB_')

  host: str = Field(default='localhost', min_length=1)
  port: int = Field(default=5432, gt=0)
  database: str = Field(default='pg-syllo-eval', min_length=1, validation_alias='DB_NAME')
  user: str = Field(default='postgres', min_length=1)
  password: str | None = None
  min_size: int = Field(default=2, ge=1)
  max_size: int = Field(default=10, ge=1)
  timeout: float = Field(default=1.0, gt=0)


class PhoenixSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='PHOENIX_')

  base_url: str = Field(default='http://localhost:6006', min_length=1)
  api_key: str | None = None
  project_id: str | None = None
  timeout: int = Field(default=1000, gt=0)
  max_retries: int = Field(default=5, ge=1)
  trace_fetch_initial_backoff_seconds: float = Field(default=5.0, gt=0)
  trace_fetch_max_backoff_seconds: float = Field(default=30.0, gt=0)
  request_id_lookup_max_attempts: int = Field(default=10, ge=1)
  request_id_lookup_initial_backoff_seconds: float = Field(default=2.0, gt=0)
  request_id_lookup_max_backoff_seconds: float = Field(default=30.0, gt=0)
  request_id_lookup_time_window_seconds: float = Field(default=600.0, gt=0)
  # Root-span attribute holding the request ID; dots address nested attributes, e.g. `metadata.request_id`.
  request_id_attribute: str = Field(default='request_id', pattern=r'^[^.]+(\.[^.]+)*$')
  request_id_excluded_root_span_names: tuple[str, ...] = ()

  @field_validator('request_id_excluded_root_span_names')
  @classmethod
  def normalize_span_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
    # Root span names are compared case-insensitively.
    return tuple(name.strip().lower() for name in names)


class OpenAIJudgeSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='OPENAI_')

  api_key: str | None = None
  base_url: str = Field(default='https://api.openai.com/v1', min_length=1)
  model: str = Field(default='gpt-5-mini', min_length=1)
  timeout_seconds: float = Field(default=30.0, gt=0)
  max_retries: int = Field(default=3, ge=1)
  use_responses_api: bool = True


class GeminiJudgeSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='GEMINI_')

  api_key: str | None = Field(default=None, validation_alias='GOOGLE_API_KEY')
  base_url: str | None = None
  model: str = Field(default='gemini-3.1-flash-lite', min_length=1)
  timeout_seconds: float = Field(default=30.0, gt=0)
  max_retries: int = Field(default=3, ge=1)


class LlmJudgeSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='LLM_JUDGE_')

  provider: Literal['openai', 'gemini'] | None = None
  max_concurrent_requests: int = Field(default=5, ge=1)
  openai: OpenAIJudgeSettings = Field(default_factory=OpenAIJudgeSettings)
  gemini: GeminiJudgeSettings = Field(default_factory=GeminiJudgeSettings)

  @field_validator('provider', mode='before')
  @classmethod
  def _normalize_provider(cls, value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


class OrbitalsSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='ORBITALS_')

  base_url: str = Field(default='https://api.orbitals.principled.app', min_length=1)
  api_key: str | None = None
  claim_extractor_model: str = Field(default='claim-extractor-pro-2605', min_length=1)
  ai_service_description: str | None = None
  timeout_seconds: float = Field(default=30.0, gt=0)


class EvaluationSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='EVALUATION_')

  max_concurrent_samples: int = Field(default=1, ge=1)
  max_concurrent_tasks: int = Field(default=10, ge=1)
  sample_trace_timeout: float | None = Field(
    default=None, gt=0, validation_alias='EVALUATION_SAMPLE_TRACE_TIMEOUT_SECONDS'
  )
  sample_compute_timeout: float | None = Field(
    default=None, gt=0, validation_alias='EVALUATION_SAMPLE_COMPUTE_TIMEOUT_SECONDS'
  )
  # Span types the built-in retrieval metrics score: the agent's final context by default, or for example the span
  # type an adapter gives each search call, to score one search at a time.
  retrieval_span_types: tuple[str, ...] = Field(default=('agent_root',), min_length=1)

  @field_validator('retrieval_span_types')
  @classmethod
  def span_types_are_named(cls, span_types: tuple[str, ...]) -> tuple[str, ...]:
    if any(not span_type.strip() for span_type in span_types):
      raise ValueError('retrieval span types must be non-empty')
    return tuple(span_type.strip() for span_type in span_types)


class Settings(EnvSettings):
  """Engine configuration consumed by ``EvaluationService``."""

  database: DatabaseSettings = Field(default_factory=DatabaseSettings)
  phoenix: PhoenixSettings = Field(default_factory=PhoenixSettings)
  llm_judge: LlmJudgeSettings = Field(default_factory=LlmJudgeSettings)
  orbitals: OrbitalsSettings = Field(default_factory=OrbitalsSettings)
  evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
