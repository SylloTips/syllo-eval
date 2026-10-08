"""The files a collection run writes, in one folder per run: ``<root>/<kind>/<run id>/``.

The service creates the run before it calls any agent, and only then is its id known: ``bind`` sets it before the run
executes, and asking for a folder earlier is an error.
"""

from pathlib import Path
from uuid import UUID


class RunOutputs:
  def __init__(self, root: Path):
    self.root = root.resolve()
    self._run_id: UUID | None = None

  def bind(self, run_id: UUID) -> None:
    self._run_id = run_id

  @property
  def run_id(self) -> UUID:
    if self._run_id is None:
      raise RuntimeError('The run has not been created yet')
    return self._run_id

  def directory(self, kind: str) -> Path:
    path = self.root / kind / str(self.run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path
