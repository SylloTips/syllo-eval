from collections.abc import Callable

from pydantic import Field

from syllo_eval.logging_utils import LoggingSettings
from syllo_eval.service import EvaluationService
from syllo_eval.settings import EnvSettings, Settings

ServiceFactory = Callable[[Settings], EvaluationService]


class AppSettings(EnvSettings):
  """Engine settings plus the process-level configuration around them."""

  engine: Settings = Field(default_factory=Settings)
  logging: LoggingSettings = Field(default_factory=LoggingSettings)

  @classmethod
  def from_env(cls) -> 'AppSettings':
    return cls(engine=Settings(), logging=LoggingSettings.from_env())


def build_evaluation_service(settings: Settings) -> EvaluationService:
  return EvaluationService(settings=settings)
