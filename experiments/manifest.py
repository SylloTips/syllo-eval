"""Append-only record of experiment steps and the evaluation runs they produced.

Each step (for example ``collect:erb/react/sonnet``) appends one JSON line per status change; the latest line of a
step is its current state, so a campaign can resume by skipping completed steps.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StepStatus(StrEnum):
  STARTED = 'started'
  COMPLETED = 'completed'
  FAILED = 'failed'


class StepRecord(BaseModel):
  model_config = ConfigDict(extra='forbid', frozen=True)

  step: str = Field(min_length=1)
  status: StepStatus
  run_ids: tuple[UUID, ...] = ()
  details: dict[str, JsonValue] = Field(default_factory=dict)
  recorded_at: datetime


class Manifest:
  def __init__(self, path: Path):
    self._path = path

  def append(
    self,
    step: str,
    status: StepStatus,
    *,
    run_ids: tuple[UUID, ...] = (),
    details: Mapping[str, Any] | None = None,
  ) -> StepRecord:
    record = StepRecord(
      step=step,
      status=status,
      run_ids=run_ids,
      details=dict(details or {}),
      recorded_at=datetime.now(timezone.utc),
    )
    self._path.parent.mkdir(parents=True, exist_ok=True)
    with self._path.open('a', encoding='utf-8') as file:
      file.write(record.model_dump_json() + '\n')
    return record

  def records(self) -> list[StepRecord]:
    if not self._path.exists():
      return []
    with self._path.open(encoding='utf-8') as file:
      return [StepRecord.model_validate_json(line) for line in file if line.strip()]

  def latest(self) -> dict[str, StepRecord]:
    """The current state of every step, keyed by step."""
    return {record.step: record for record in self.records()}

  def is_completed(self, step: str) -> bool:
    record = self.latest().get(step)
    return record is not None and record.status is StepStatus.COMPLETED
