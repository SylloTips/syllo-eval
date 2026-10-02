"""Validated experiment configuration: the models and agent configurations of Section 5.1."""

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'

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


class ModelsConfig(_ConfigModel):
  agents: dict[str, ModelRef] = Field(min_length=1)
  customer_simulator: ModelRef
  judge: JudgeConfig


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


class ExperimentConfig(_ConfigModel):
  models: ModelsConfig
  configurations: tuple[Configuration, ...] = Field(min_length=1)

  @model_validator(mode='after')
  def _check_references(self) -> 'ExperimentConfig':
    duplicate_ids = [key for key, count in Counter(c.id for c in self.configurations).items() if count > 1]
    if duplicate_ids:
      raise ValueError(f'Duplicate configuration ids: {duplicate_ids}')
    unknown_models = sorted({c.model for c in self.configurations} - set(self.models.agents))
    if unknown_models:
      raise ValueError(f'Configurations use undefined agent models: {unknown_models}')
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
  )


def _read_yaml(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return yaml.safe_load(file)
