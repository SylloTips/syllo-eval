from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

from syllo_eval.evaluation.metric_support.retrieved_context import normalize_relevant_ids
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import (
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  GroundTruthKey,
  MetricComputation,
)

from config import Configuration
from judge import JUDGE_OUTCOMES

REPORTED_AGENTS = frozenset({'react', 'smolagents', 'deepagents'})
_PAGE_SIZE = 500


@dataclass(frozen=True)
class JudgeRun:
  run_ids: tuple[UUID, ...]
  source_run_id: UUID
  configuration: Configuration
  arm: str
  repetition: int

  @property
  def reported(self) -> bool:
    return self.configuration.agent in REPORTED_AGENTS and not self.configuration.degradation


async def finished_runs(db_manager: DatabaseManager) -> list[EvaluationRun]:
  runs: list[EvaluationRun] = []
  async with UnitOfWork(db_manager) as uow:
    while page := await uow.evaluation_runs.list_runs(limit=_PAGE_SIZE, offset=len(runs)):
      runs.extend(page)
  return [run for run in runs if run.status is not EvaluationStatus.RUNNING]


def judge_runs(
  runs: Sequence[EvaluationRun], configurations: Sequence[Configuration], arms: Collection[str]
) -> list[JudgeRun]:
  by_id = {configuration.id: configuration for configuration in configurations}
  chains: defaultdict[tuple[str, str, int], list[EvaluationRun]] = defaultdict(list)
  sources: defaultdict[tuple[str, str, int], set[UUID]] = defaultdict(set)
  for run in runs:
    metadata = run.metadata or {}
    arm, step, source = metadata.get('arm'), metadata.get('step'), run.source_run_id
    if source is None or not isinstance(arm, str) or arm not in arms:
      continue
    if isinstance(step, str) and step.endswith(':pilot'):
      continue
    configuration_id, repetition = metadata.get('configuration'), metadata.get('repetition')
    if not isinstance(configuration_id, str) or not isinstance(repetition, int):
      raise ValueError(f'run {run.id} has no configuration and repetition in its metadata')
    if configuration_id not in by_id:
      raise ValueError(f'run {run.id} names configuration {configuration_id!r}, which configs/ does not list')
    chains[(configuration_id, arm, repetition)].append(run)
    sources[(configuration_id, arm, repetition)].add(source)
  found: list[JudgeRun] = []
  for key, chain in chains.items():
    if len(sources[key]) > 1:
      raise ValueError(f'{key} scores several source runs: {sorted(map(str, sources[key]))}')
    chain.sort(key=lambda run: (run.start_time, run.id))
    found.append(JudgeRun(tuple(run.id for run in chain), next(iter(sources[key])), by_id[key[0]], key[1], key[2]))
  order = {configuration.id: index for index, configuration in enumerate(configurations)}
  return sorted(found, key=lambda run: (order[run.configuration.id], run.arm, run.repetition))


def latest_attempts(
  attempts: Sequence[tuple[Sequence[EvaluationRunSample], Sequence[MetricComputation]]], metrics: Collection[str]
) -> tuple[list[MetricComputation], dict[UUID, UUID]]:
  latest: dict[tuple[UUID, str], list[MetricComputation]] = {}
  sample_ids: dict[UUID, UUID] = {}
  for run_samples, computations in attempts:
    completed = {
      run_sample.id: run_sample.sample_id
      for run_sample in run_samples
      if run_sample.status is EvaluationSampleStatus.COMPLETED
    }
    scored: defaultdict[tuple[UUID, str], list[MetricComputation]] = defaultdict(list)
    for computation in computations:
      if computation.metric in metrics and computation.evaluation_run_sample_id in completed:
        scored[(completed[computation.evaluation_run_sample_id], computation.metric)].append(computation)
    latest.update(scored)
    sample_ids.update(completed)
  return [computation for computations in latest.values() for computation in computations], sample_ids


async def load_attempts(
  db_manager: DatabaseManager, run_ids: Sequence[UUID], metrics: Collection[str]
) -> tuple[list[MetricComputation], dict[UUID, UUID]]:
  async with UnitOfWork(db_manager) as uow:
    attempts = [
      (
        await uow.evaluation_run_samples.list_by_evaluation_run(run_id),
        await uow.metric_computations.list_by_evaluation_run(run_id),
      )
      for run_id in run_ids
    ]
  return latest_attempts(attempts, metrics)


async def gold_documents(db_manager: DatabaseManager) -> dict[UUID, frozenset[str]]:
  async with UnitOfWork(db_manager) as uow:
    labels = await uow.ground_truths.list_by_key(GroundTruthKey.RELEVANT_DOCUMENT_IDS.value)
  return {
    label.sample_id: frozenset(normalize_relevant_ids(label.ground_truth_value.get('relevant_ids') or []))
    for label in labels
  }


def rounded(value: float | None) -> str:
  return '–' if value is None else str(Decimal(repr(value)).quantize(Decimal('0.001'), ROUND_HALF_UP))


def require_judge_outcome(computation: MetricComputation) -> None:
  failure = (computation.metadata or {}).get('failure')
  if failure not in JUDGE_OUTCOMES:
    raise ValueError(f'unit {computation.id} failed with {failure!r}; score it again before the analysis')
