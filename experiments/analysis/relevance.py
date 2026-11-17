from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.ndcg_at_10 import NdcgAt10DocumentMetric
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import MetricComputation, MetricComputationStatus

from ablation_metrics import ContextualPrecisionSC
from analysis.runs import (
  JudgeRun,
  finished_runs,
  gold_documents,
  judge_runs,
  load_attempts,
  require_judge_outcome,
  rounded,
)
from config import Configuration
from deepeval_baseline import DeepEvalContextualPrecision

ARMS = ('main', 'sc', 'deepeval')
CP_METRICS = frozenset(
  metric.metric_name
  for metric in (ContextualPrecisionDocumentJudgeMetric, ContextualPrecisionSC, DeepEvalContextualPrecision)
)


@dataclass(frozen=True)
class Decision:
  sample_id: UUID
  item_id: str
  accepted: bool
  gold: bool
  judged: bool = True


def _rankings(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
  return metadata.get('rankings') or [metadata]


def decisions(
  computations: Sequence[MetricComputation],
  sample_ids: Mapping[UUID, UUID],
  span_starts: Mapping[str, datetime],
  gold: Mapping[UUID, frozenset[str]],
  source_rankings: Mapping[str, list[str]],
) -> list[Decision]:
  ordered: list[tuple[tuple[UUID, datetime, str, int], str, bool, bool]] = []
  for computation in computations:
    sample_id = sample_ids[computation.evaluation_run_sample_id]
    if (
      computation.metric not in CP_METRICS
      or computation.status is MetricComputationStatus.SKIPPED
      or not gold.get(sample_id)
    ):
      continue
    span_id = computation.span_ids[0]
    if computation.status is MetricComputationStatus.COMPLETED:
      verdicts = [
        (result['retrieved_id'], result['relevant'])
        for ranking in _rankings(computation.metadata or {})
        for result in ranking['rank_results']
      ]
    else:
      require_judge_outcome(computation)
      if span_id not in source_rankings:
        raise ValueError(f'failed unit {computation.id} has no NDCG@10 ranking in the source run')
      verdicts = [(document_id, document_id not in gold[sample_id]) for document_id in source_rankings[span_id]]
    ordered.extend(
      ((sample_id, span_starts[span_id], span_id, rank), document_id, accepted, document_id in gold[sample_id])
      for rank, (document_id, accepted) in enumerate(verdicts)
    )
  first: dict[tuple[UUID, str], Decision] = {}
  for key, document_id, accepted, is_gold in sorted(ordered, key=lambda item: item[0]):
    first.setdefault((key[0], document_id), Decision(key[0], document_id, accepted, is_gold))
  return list(first.values())


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
  if not pairs:
    return None
  observed = sum(a == b for a, b in pairs) / len(pairs)
  share_a, share_b = sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs)
  chance = share_a * share_b + (1 - share_a) * (1 - share_b)
  return None if chance == 1 else (observed - chance) / (1 - chance)


def summary(decisions: Sequence[Decision]) -> dict[str, Any]:
  gold = [decision.accepted for decision in decisions if decision.gold]
  other = [decision.accepted for decision in decisions if not decision.gold]
  return {
    'questions': len({decision.sample_id for decision in decisions}),
    'gold': len(gold),
    'gold_accepted': sum(gold) / len(gold) if gold else None,
    'other': len(other),
    'other_accepted': sum(other) / len(other) if other else None,
    'kappa': cohen_kappa([(decision.accepted, decision.gold) for decision in decisions]),
    'unjudged': sum(not decision.judged for decision in decisions),
  }


async def _run_decisions(
  db_manager: DatabaseManager, run: JudgeRun, gold: Mapping[UUID, frozenset[str]]
) -> list[Decision]:
  computations, sample_ids = await load_attempts(db_manager, run.run_ids, CP_METRICS)
  source: list[MetricComputation] = []
  if any(c.status is MetricComputationStatus.FAILED for c in computations):
    async with UnitOfWork(db_manager) as uow:
      source = await uow.metric_computations.list_by_evaluation_run(run.source_run_id)
  span_ids = list({c.span_ids[0] for c in computations if c.span_ids})
  async with db_manager.get_async_connection() as conn:
    cursor = await conn.execute('SELECT external_id, start_time FROM span WHERE external_id = ANY(%s)', (span_ids,))
    span_starts = dict(await cursor.fetchall())
  source_rankings = {
    c.span_ids[0]: [document_id for ranking in _rankings(c.metadata or {}) for document_id in ranking['ranked_ids']]
    for c in source
    if c.metric == NdcgAt10DocumentMetric.metric_name and c.status is MetricComputationStatus.COMPLETED
  }
  return decisions(computations, sample_ids, span_starts, gold, source_rankings)


async def decision_rows(
  runs: Sequence[JudgeRun], run_decisions: Callable[[JudgeRun], Awaitable[list[Decision]]]
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  pooled: defaultdict[tuple[str, str, int], list[Decision]] = defaultdict(list)
  for run in runs:
    found = await run_decisions(run)
    rows.append(
      {
        'configuration': run.configuration.id,
        'arm': run.arm,
        'repetition': run.repetition,
        'run_ids': [str(run_id) for run_id in run.run_ids],
        **summary(found),
      }
    )
    pooled[(run.configuration.benchmark, run.arm, run.repetition)].extend(found)
  rows.extend(
    {'configuration': f'{benchmark}/reported', 'arm': arm, 'repetition': repetition, **summary(pooled_decisions)}
    for (benchmark, arm, repetition), pooled_decisions in pooled.items()
  )
  return rows


async def relevance(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  runs = judge_runs(await finished_runs(db_manager), configurations, arms)
  gold = await gold_documents(db_manager)
  runs = [run for run in runs if run.reported and run.configuration.benchmark != 'tau2']
  return await decision_rows(runs, lambda run: _run_decisions(db_manager, run, gold))


def decision_table(
  rows: Sequence[Mapping[str, Any]], gold: str = 'Gold', other: str = 'Other', unjudged: str | None = None
) -> list[str]:
  extra = f' {unjudged} |' if unjudged else ''
  lines = [
    f'| Configuration | Arm | Rep | Questions | {gold} | {gold} accepted | {other} | {other} accepted |{extra} κ |',
    '|---|---|--:|--:|--:|--:|--:|--:|' + ('--:|' if unjudged else '') + '--:|',
  ]
  for row in rows:
    cell = f' {row["unjudged"]} |' if unjudged else ''
    lines.append(
      f'| {row["configuration"]} | {row["arm"]} | {row["repetition"]} | {row["questions"]} | {row["gold"]} | '
      f'{rounded(row["gold_accepted"])} | {row["other"]} | {rounded(row["other_accepted"])} |{cell} '
      f'{rounded(row["kappa"])} |'
    )
  return lines
