"""Validated experiment configuration: the models and agent configurations of Section 5.1."""

import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from syllo_eval.settings import load_settings_env

EXPERIMENTS_DIR = Path(__file__).resolve().parent
CONFIG_DIR = EXPERIMENTS_DIR / 'configs'
DATA_DIR = EXPERIMENTS_DIR / 'data'

Benchmark = Literal['erb', 'wixqa', 'tau2']
AgentStack = Literal['react', 'smolagents', 'odr', 'tau2-llm-agent']


class _ConfigModel(BaseModel):
  model_config = ConfigDict(extra='forbid', frozen=True)


class ModelRef(_ConfigModel):
  provider: str = Field(min_length=1)
  model: str = Field(min_length=1)
  label: str = Field(min_length=1)


class JudgeConfig(_ConfigModel):
  provider: Literal['gemini']
  model: str = Field(min_length=1)
  label: str = Field(min_length=1)
  temperature: float = Field(ge=0.0)
  thinking_level: Literal['minimal', 'low', 'medium', 'high'] | None = None
  timeout_seconds: float = Field(gt=0)
  max_retries: int = Field(ge=1)
  max_concurrent_requests: int = Field(ge=1)
  # Output tokens the judge model can return in one response; it caps the budget of a single-call ablation.
  output_token_limit: int = Field(gt=0)


class EmbeddingConfig(_ConfigModel):
  provider: Literal['azure_foundry']
  # The Foundry deployment name.
  model: str = Field(min_length=1)
  label: str = Field(min_length=1)
  output_dimension: int = Field(gt=0)
  timeout_seconds: float = Field(gt=0)
  max_retries: int = Field(ge=1)
  max_concurrent_requests: int = Field(ge=1)
  # The deployment's rate limit, and the estimated tokens a single request may carry.
  tokens_per_minute: int = Field(gt=0)
  max_request_tokens: int = Field(gt=0)

  @model_validator(mode='after')
  def _check_request_budget(self) -> 'EmbeddingConfig':
    if self.max_request_tokens > self.tokens_per_minute:
      raise ValueError('max_request_tokens cannot exceed tokens_per_minute')
    return self


class ModelsConfig(_ConfigModel):
  agents: dict[str, ModelRef] = Field(min_length=1)
  customer_simulator: ModelRef
  judge: JudgeConfig
  embedding: EmbeddingConfig


class Tau2Config(_ConfigModel):
  """How tau2-bench runs its agent; see ``configs/tau2.yaml``."""

  # litellm arguments, per agent model key and for the customer simulator.
  agent_llm_args: dict[str, dict[str, JsonValue]]
  customer_llm_args: dict[str, JsonValue]
  max_steps: int = Field(gt=0)
  max_errors: int = Field(gt=0)
  base_seed: int
  cost_map_url: str = Field(pattern=r'^https://\S+/[0-9a-f]{40}/\S+\.json$')

  def trial_seed(self, trial: int) -> int:
    """The seed of 1-based ``trial``: tau2's runner draws one seed per trial from the base seed."""
    draws = random.Random(self.base_seed)
    return [draws.randint(0, 1_000_000) for _ in range(trial)][-1]


class OdrConfig(_ConfigModel):
  """How Open Deep Research runs; see ``configs/odr.yaml``. The names are ODR's own settings."""

  # Web search stays off: the search tool is the only source.
  search_api: Literal['none']
  allow_clarification: bool
  max_concurrent_research_units: int = Field(gt=0)
  max_researcher_iterations: int = Field(gt=0)
  max_react_tool_calls: int = Field(gt=0)
  max_structured_output_retries: int = Field(gt=0)
  research_model_max_tokens: int = Field(gt=0)
  compression_model_max_tokens: int = Field(gt=0)
  final_report_model_max_tokens: int = Field(gt=0)
  mcp_prompt: str = Field(min_length=1)


class Configuration(_ConfigModel):
  id: str = Field(min_length=1)
  benchmark: Benchmark
  agent: AgentStack
  model: str = Field(min_length=1)
  degradation: float = Field(default=0.0, ge=0.0, lt=1.0)
  trial: int | None = Field(default=None, ge=1)

  @model_validator(mode='after')
  def _check_benchmark_shape(self) -> 'Configuration':
    is_tau2 = self.benchmark == 'tau2'
    if is_tau2 != (self.agent == 'tau2-llm-agent'):
      raise ValueError(f'{self.id}: tau2 runs only the tau2 agent, and the tau2 agent only runs tau2')
    if is_tau2 != (self.trial is not None):
      raise ValueError(f'{self.id}: tau2 configurations need a trial, and only they may have one')
    if is_tau2 and self.degradation:
      raise ValueError(f'{self.id}: degradation applies to knowledge-base search, which tau2 does not use')
    return self

  @property
  def version_tag(self) -> str:
    """Syllo-eval agent version tag; the agent name is the stack."""
    parts = [self.model]
    if self.degradation:
      parts.append(f'f{self.degradation:g}')
    if self.trial is not None:
      parts.append(f'trial{self.trial}')
    return '-'.join(parts)


class PinnedFile(_ConfigModel):
  path: str = Field(min_length=1)
  size: int = Field(gt=0)
  sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class BenchmarkSource(_ConfigModel):
  dataset_name: str = Field(min_length=1)
  expected_samples: int = Field(gt=0)
  # Files are fetched from f'{base_url}/{path}'; the URL must name a full commit, so the bytes can never change.
  base_url: str = Field(pattern=r'^https://\S+/[0-9a-f]{40}$')
  files: dict[str, PinnedFile] = Field(min_length=1)


class ExperimentConfig(_ConfigModel):
  models: ModelsConfig
  configurations: tuple[Configuration, ...] = Field(min_length=1)
  benchmarks: dict[Benchmark, BenchmarkSource]
  tau2: Tau2Config
  odr: OdrConfig

  @model_validator(mode='after')
  def _check_references(self) -> 'ExperimentConfig':
    missing_benchmarks = sorted({c.benchmark for c in self.configurations} - set(self.benchmarks))
    if missing_benchmarks:
      raise ValueError(f'Configurations use benchmarks without a pinned source: {missing_benchmarks}')
    duplicate_ids = [key for key, count in Counter(c.id for c in self.configurations).items() if count > 1]
    if duplicate_ids:
      raise ValueError(f'Duplicate configuration ids: {duplicate_ids}')
    unknown_models = sorted({c.model for c in self.configurations} - set(self.models.agents))
    if unknown_models:
      raise ValueError(f'Configurations use undefined agent models: {unknown_models}')
    tau2_models = {c.model for c in self.configurations if c.benchmark == 'tau2'}
    missing_args = sorted(tau2_models - set(self.tau2.agent_llm_args))
    if missing_args:
      raise ValueError(f'tau2 configurations use agent models without llm args in tau2.yaml: {missing_args}')
    # Agent rows are unique per (name, version tag), and each run evaluates one agent on one dataset.
    runs = Counter((c.benchmark, c.agent, c.version_tag) for c in self.configurations)
    clashing = [run for run, count in runs.items() if count > 1]
    if clashing:
      raise ValueError(f'Configurations map to the same agent version on one benchmark: {clashing}')
    return self

  def configuration(self, configuration_id: str) -> Configuration:
    for configuration in self.configurations:
      if configuration.id == configuration_id:
        return configuration
    raise KeyError(f'Unknown configuration {configuration_id!r}')


def load_config(config_dir: Path = CONFIG_DIR) -> ExperimentConfig:
  return ExperimentConfig(
    models=_read_yaml(config_dir / 'models.yaml'),
    configurations=_read_yaml(config_dir / 'configurations.yaml')['configurations'],
    benchmarks=_read_yaml(config_dir / 'benchmarks.yaml'),
    tau2=_read_yaml(config_dir / 'tau2.yaml'),
    odr=_read_yaml(config_dir / 'odr.yaml'),
  )


def _read_yaml(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return yaml.safe_load(file)


def load_environment() -> None:
  """Load settings for entry points: ``experiments/.env`` first, so it wins over the repository's ``.env``."""
  load_settings_env(dotenv_path=EXPERIMENTS_DIR / '.env')
  load_settings_env(dotenv_path=EXPERIMENTS_DIR.parent / '.env')
