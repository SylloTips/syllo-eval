import contextlib
import logging
import os
import sys
from contextvars import ContextVar
from typing import Any, Iterator

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from syllo_eval.settings import EnvSettings

_PLAIN_FORMAT = '%(asctime)s %(levelname)s %(name)s - %(context)s%(message)s'
_NOISY_LOGGERS = ('httpx', 'httpcore', 'google_genai.models', 'urllib3', 'openai')

_run_id_var: ContextVar[str | None] = ContextVar('log_run_id', default=None)
_sample_id_var: ContextVar[str | None] = ContextVar('log_sample_id', default=None)

_RESET = '\033[0m'
_DIM = '\033[2m'
_CYAN = '\033[36m'
_GREEN = '\033[32m'
_YELLOW = '\033[33m'
_RED = '\033[31m'
_BOLD_RED = '\033[1;31m'
_GREY = '\033[90m'

_LEVEL_COLORS = {
  logging.DEBUG: _DIM,
  logging.INFO: _GREEN,
  logging.WARNING: _YELLOW,
  logging.ERROR: _RED,
  logging.CRITICAL: _BOLD_RED,
}


@contextlib.contextmanager
def bind_run_id(run_id: object) -> Iterator[None]:
  token = _run_id_var.set(str(run_id))
  try:
    yield
  finally:
    _run_id_var.reset(token)


@contextlib.contextmanager
def bind_sample_id(sample_id: object) -> Iterator[None]:
  token = _sample_id_var.set(str(sample_id))
  try:
    yield
  finally:
    _sample_id_var.reset(token)


class _ContextFilter(logging.Filter):
  def __init__(self, id_length: int | None) -> None:
    super().__init__()
    self._id_length = id_length

  def filter(self, record: logging.LogRecord) -> bool:
    parts: list[str] = []
    run_id = _run_id_var.get()
    if run_id:
      parts.append(f'run={self._truncate(run_id)}')
    sample_id = _sample_id_var.get()
    if sample_id:
      parts.append(f'sample={self._truncate(sample_id)}')
    record.context = f'[{" ".join(parts)}] ' if parts else ''
    return True

  def _truncate(self, value: str) -> str:
    return value if self._id_length is None else value[: self._id_length]


class _ColorFormatter(logging.Formatter):
  def __init__(self) -> None:
    super().__init__('%(message)s')

  def format(self, record: logging.LogRecord) -> str:
    level_color = _LEVEL_COLORS.get(record.levelno, '')
    name_color = _CYAN if record.name.startswith('syllo_eval.') else _GREY
    asctime = self.formatTime(record)
    message = super().format(record)
    context = getattr(record, 'context', '')
    return (
      f'{_DIM}{asctime}{_RESET} '
      f'{level_color}{record.levelname:<8}{_RESET} '
      f'{name_color}{record.name}{_RESET} - '
      f'{_CYAN}{context}{_RESET}{message}'
    )


_DEFAULT_ID_LENGTH = 8


def _resolve_log_level(value: str | int | None) -> int:
  if isinstance(value, int):
    return value
  if value is None:
    return logging.INFO
  return getattr(logging, value.strip().upper(), logging.INFO)


def _resolve_use_color() -> bool:
  if (os.environ.get('NO_COLOR') or '').strip():
    return False
  if (os.environ.get('FORCE_COLOR') or '').strip():
    return True
  return sys.stderr.isatty()


class LoggingSettings(EnvSettings):
  """Process-level logging configuration.

  ``LOG_LEVEL`` and ``LOG_ID_LENGTH`` are read declaratively; only ``use_color`` needs
  ``from_env``, since it depends on whether stderr is a TTY and on the presence-based
  ``NO_COLOR``/``FORCE_COLOR`` conventions.
  """

  model_config = SettingsConfigDict(env_prefix='LOG_')

  level: int = Field(default=logging.INFO)
  use_color: bool = True
  id_length: int | None = Field(default=_DEFAULT_ID_LENGTH)

  @field_validator('level', mode='before')
  @classmethod
  def _parse_level(cls, value: Any) -> int:
    return _resolve_log_level(value)

  @field_validator('id_length', mode='before')
  @classmethod
  def _parse_id_length(cls, value: Any) -> Any:
    if not isinstance(value, str):
      return value
    if value.lower() == 'full':
      return None
    try:
      parsed = int(value)
    except ValueError:
      return _DEFAULT_ID_LENGTH
    return parsed if parsed > 0 else _DEFAULT_ID_LENGTH

  @classmethod
  def from_env(cls, level_override: str | int | None = None) -> 'LoggingSettings':
    overrides: dict[str, Any] = {'use_color': _resolve_use_color()}
    if level_override is not None:
      overrides['level'] = level_override
    return cls(**overrides)


def configure_logging(settings: LoggingSettings | None = None) -> None:
  resolved = settings or LoggingSettings.from_env()
  root_logger = logging.getLogger()
  root_logger.setLevel(resolved.level)

  if root_logger.handlers:
    return

  handler = logging.StreamHandler()
  handler.addFilter(_ContextFilter(resolved.id_length))
  handler.setFormatter(_ColorFormatter() if resolved.use_color else logging.Formatter(_PLAIN_FORMAT))
  root_logger.addHandler(handler)

  for noisy in _NOISY_LOGGERS:
    logging.getLogger(noisy).setLevel(logging.WARNING)
