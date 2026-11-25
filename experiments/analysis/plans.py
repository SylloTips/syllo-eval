import json
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import UUID

from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import GroundTruthKey, MetricComputationStatus

from ablation_metrics import PlanCorrectnessWT
from analysis.runs import JudgeRun, finished_runs, judge_runs, load_attempts, require_judge_outcome, rounded
from config import Configuration

ARMS = ('main', 'wt')
PC_METRICS = frozenset({PlanCorrectnessJudgeMetric.metric_name, PlanCorrectnessWT.metric_name})
COLUMNS = ('trajectories', 'rewarded', 'pc_auroc', 'matching_auroc', 'deviating', 'deviating_pc', 'deviating_matching')


@dataclass(frozen=True)
class Trajectory:
  pc: float
  matching: float
  rewarded: bool
  follows_reference: bool


def call(operation: str, arguments: Any) -> tuple[str, str]:
  return operation, json.dumps(arguments, sort_keys=True)


def matching(executed: Sequence[tuple[str, str]], expected: Sequence[tuple[str, str]]) -> float:
  if not executed and not expected:
    return 1.0
  previous = list(range(len(expected) + 1))
  for i, step in enumerate(executed, 1):
    current = [i]
    for j, reference in enumerate(expected, 1):
      current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (step != reference)))
    previous = current
  return 1 - previous[-1] / max(len(executed), len(expected))


def auroc(points: Sequence[tuple[float, bool]]) -> float | None:
  positives = [score for score, rewarded in points if rewarded]
  negatives = [score for score, rewarded in points if not rewarded]
  if not positives or not negatives:
    return None
  return sum((p > n) + (p == n) / 2 for p in positives for n in negatives) / (len(positives) * len(negatives))


def summary(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
  deviating = [t for t in trajectories if t.rewarded and not t.follows_reference]
  return {
    'trajectories': len(trajectories),
    'rewarded': sum(t.rewarded for t in trajectories),
    'pc_auroc': auroc([(t.pc, t.rewarded) for t in trajectories]),
    'matching_auroc': auroc([(t.matching, t.rewarded) for t in trajectories]),
    'deviating': len(deviating),
    'deviating_pc': mean(t.pc for t in deviating) if deviating else None,
    'deviating_matching': mean(t.matching for t in deviating) if deviating else None,
  }


def read_rewards(path: Path) -> dict[UUID, tuple[bool, bool]]:
  latest: dict[UUID, dict[str, Any]] = {}
  for line in path.read_text(encoding='utf-8').splitlines():
    record = json.loads(line)
    sample_id = UUID(record['sample_id'])
    if sample_id not in latest or record['attempt'] > latest[sample_id]['attempt']:
      latest[sample_id] = record
  return {
    sample_id: (
      record['reward'] == 1,
      all(check['action_match'] for check in record['reward_info'].get('action_checks') or ()),
    )
    for sample_id, record in latest.items()
  }


async def _source(
  db_manager: DatabaseManager, run_id: UUID, rewards_dir: Path, expected: Mapping[UUID, list[tuple[str, str]]]
) -> dict[UUID, tuple[float, bool, bool, str]]:
  rewards = read_rewards(rewards_dir / str(run_id) / 'rewards.jsonl')
  found: dict[UUID, tuple[float, bool, bool, str]] = {}
  async with UnitOfWork(db_manager) as uow:
    for run_sample in await uow.evaluation_run_samples.list_by_evaluation_run(run_id):
      if run_sample.trace_id is None or run_sample.sample_id not in expected:
        continue
      (root,) = await uow.spans.list_by_span_type(run_sample.trace_id, 'agent_root')
      planning = root.semantics.planning
      executed = [call(step.operation, step.input) for step in (planning.executed_steps if planning else None) or ()]
      rewarded, follows = rewards[run_sample.sample_id]
      found[run_sample.sample_id] = (
        matching(executed, expected[run_sample.sample_id]),
        rewarded,
        follows,
        run_sample.trace_id,
      )
  return found


async def trajectories(
  db_manager: DatabaseManager,
  configurations: Sequence[Configuration],
  arms: Collection[str],
  rewards_dir: Path,
  cohort_arms: Collection[str],
) -> tuple[list[tuple[JudgeRun, list[tuple[str, Trajectory]]]], set[str]]:
  runs = [
    run
    for run in judge_runs(await finished_runs(db_manager), configurations, {*arms, *cohort_arms})
    if run.configuration.benchmark == 'tau2'
  ]
  async with UnitOfWork(db_manager) as uow:
    rows = await uow.ground_truths.list_by_key(GroundTruthKey.EXPECTED_PLAN.value)
  expected = {
    row.sample_id: [call(step['operation'], step.get('parameters')) for step in row.ground_truth_value['expected_plan']]
    for row in rows
  }
  sources: dict[UUID, dict[UUID, tuple[float, bool, bool, str]]] = {}
  for run in runs:
    if run.source_run_id not in sources:
      sources[run.source_run_id] = await _source(db_manager, run.source_run_id, rewards_dir, expected)
  found: list[tuple[JudgeRun, list[tuple[str, Trajectory]]]] = []
  for run in runs:
    if run.arm not in arms:
      continue
    scored: list[tuple[str, Trajectory]] = []
    computations, sample_ids = await load_attempts(db_manager, run.run_ids, PC_METRICS)
    for computation in computations:
      if computation.status is MetricComputationStatus.FAILED:
        require_judge_outcome(computation)
      if computation.status is not MetricComputationStatus.COMPLETED or computation.score is None:
        continue
      score, rewarded, follows, trace = sources[run.source_run_id][sample_ids[computation.evaluation_run_sample_id]]
      scored.append((trace, Trajectory(computation.score, score, rewarded, follows)))
    found.append((run, scored))
  return found, {trace for source in sources.values() for *_, trace in source.values()}


async def plans(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str], rewards_dir: Path
) -> list[dict[str, Any]]:
  groups: defaultdict[tuple[str, str, int], list[Trajectory]] = defaultdict(list)
  for run, scored in (await trajectories(db_manager, configurations, arms, rewards_dir, arms))[0]:
    for _, trajectory in scored:
      for model in (run.configuration.model, 'pooled'):
        groups[(model, run.arm, run.repetition)].append(trajectory)
  return [
    {'configuration': f'tau2/{model}', 'arm': arm, 'repetition': repetition, **summary(found)}
    for (model, arm, repetition), found in sorted(
      groups.items(), key=lambda item: (*item[0][1:], item[0][0] == 'pooled', item[0][0])
    )
  ]


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines = [
    '| Configuration | Arm | Rep | Trajectories | Rewarded | AUROC PC | AUROC matching | Deviating | PC on deviating '
    '| Matching on deviating |',
    '|---|---|--:|--:|--:|--:|--:|--:|--:|--:|',
  ]
  for row in rows:
    cells = [str(row[c]) if isinstance(row[c], int) else rounded(row[c]) for c in COLUMNS]
    lines.append(f'| {row["configuration"]} | {row["arm"]} | {row["repetition"]} | ' + ' | '.join(cells) + ' |')
  return lines
