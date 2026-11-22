from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from statistics import mean
from typing import Any
from uuid import UUID

from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.ndcg_at_10 import NdcgAt10DocumentMetric
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.model import MetricComputation, MetricComputationStatus

from ablation_metrics import (
  AnswerCorrectnessWT,
  GoldClaimsContextualRecall,
  GoldClaimsContextualRecallSC,
  OrbitalsClaimsContextualRecall,
  PlanCorrectnessWT,
  RetrievedSetRecall,
)
from analysis.relevance import CP_METRICS
from analysis.runs import (
  REPORTED_AGENTS,
  JudgeRun,
  finished_runs,
  gold_documents,
  judge_runs,
  load_attempts,
  require_judge_outcome,
  rounded,
)
from config import Configuration
from deepeval_baseline import DeepEvalAnswerCorrectness, DeepEvalGoldClaimsContextualRecall

ARMS = ('main', 'sc', 'wt', 'deepeval')
COLUMNS = ('PC', 'AC', 'CP', 'CR', 'R', 'R on CR', 'NDCG@10')
LABELS = {
  **{metric.metric_name: 'PC' for metric in (PlanCorrectnessJudgeMetric, PlanCorrectnessWT)},
  **{
    metric.metric_name: 'AC'
    for metric in (AnswerCorrectnessJudgeMetric, AnswerCorrectnessWT, DeepEvalAnswerCorrectness)
  },
  **{metric: 'CP' for metric in CP_METRICS},
  **{
    metric.metric_name: 'CR'
    for metric in (
      GoldClaimsContextualRecall,
      OrbitalsClaimsContextualRecall,
      GoldClaimsContextualRecallSC,
      DeepEvalGoldClaimsContextualRecall,
    )
  },
}
SOURCE_LABELS = {RetrievedSetRecall.metric_name: 'R', NdcgAt10DocumentMetric.metric_name: 'NDCG@10'}
RETRIEVAL = frozenset({'CP', 'CR', 'R', 'NDCG@10'})
RANKING = frozenset({'CP', 'NDCG@10'})
NO_SEARCH = 'No spans found'
NO_ANSWER = 'Trace does not provide an answer'
UNAVAILABLE = 'Document retrieval unavailable'


def question_scores(
  computations: Sequence[MetricComputation],
  sample_ids: Mapping[UUID, UUID],
  labels: Mapping[str, str],
  gold: Mapping[UUID, frozenset[str]],
) -> dict[str, dict[UUID, float]]:
  units: defaultdict[tuple[str, UUID], list[MetricComputation]] = defaultdict(list)
  for computation in computations:
    if computation.metric in labels:
      units[(labels[computation.metric], sample_ids[computation.evaluation_run_sample_id])].append(computation)
  unavailable = {
    sample_id
    for (label, sample_id), group in units.items()
    if label in RETRIEVAL
    and any(
      (c.error_message or '').startswith(UNAVAILABLE) for c in group if c.status is MetricComputationStatus.SKIPPED
    )
  }
  scores: defaultdict[str, dict[UUID, float]] = defaultdict(dict)
  for (label, sample_id), group in units.items():
    if (label in RETRIEVAL and sample_id in unavailable) or (label == 'NDCG@10' and not gold.get(sample_id)):
      continue
    values: list[float] = []
    for computation in group:
      if computation.status is MetricComputationStatus.COMPLETED and computation.score is not None:
        values.append(computation.score)
      elif computation.status is MetricComputationStatus.FAILED:
        require_judge_outcome(computation)
    skips = [c.error_message or '' for c in group if c.status is MetricComputationStatus.SKIPPED]
    if values:
      scores[label][sample_id] = mean(values)
    elif any(skip.startswith(NO_SEARCH if label in RANKING else NO_ANSWER) for skip in skips):
      scores[label][sample_id] = 0.0
  return dict(scores)


def with_r_on_cr(scores: Mapping[str, Mapping[UUID, float]]) -> dict[str, Mapping[UUID, float]]:
  cr = scores.get('CR', {})
  return {**scores, 'R on CR': {q: s for q, s in scores.get('R', {}).items() if q in cr}}


def summary(scores: Mapping[str, Mapping[UUID, float]]) -> dict[str, Any]:
  scores = with_r_on_cr(scores)
  return {
    key: value
    for label in COLUMNS
    for key, value in (
      (label, mean(scores[label].values()) if scores.get(label) else None),
      (f'{label} questions', len(scores.get(label, {}))),
    )
  }


async def run_scores(
  db_manager: DatabaseManager, run: JudgeRun, gold: Mapping[UUID, frozenset[str]]
) -> dict[str, dict[UUID, float]]:
  computations, sample_ids = await load_attempts(db_manager, run.run_ids, LABELS)
  judged = set(sample_ids.values())
  source, source_sample_ids = await load_attempts(db_manager, (run.source_run_id,), SOURCE_LABELS)
  source = [c for c in source if source_sample_ids[c.evaluation_run_sample_id] in judged]
  return question_scores(
    [*computations, *source], {**sample_ids, **source_sample_ids}, {**LABELS, **SOURCE_LABELS}, gold
  )


async def scores(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  gold = await gold_documents(db_manager)
  rows: list[dict[str, Any]] = []
  for run in judge_runs(await finished_runs(db_manager), configurations, arms):
    if run.configuration.benchmark != 'tau2' and run.configuration.agent not in REPORTED_AGENTS:
      continue
    rows.append(
      {
        'configuration': run.configuration.id,
        'arm': run.arm,
        'repetition': run.repetition,
        'run_ids': [str(run_id) for run_id in run.run_ids],
        **summary(await run_scores(db_manager, run, gold)),
      }
    )
  return rows


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines = ['| Configuration | Arm | Rep | ' + ' | '.join(COLUMNS) + ' |', '|---|---|--:|' + '--:|' * len(COLUMNS)]
  for row in rows:
    cells = ['–' if row[label] is None else f'{rounded(row[label])} ({row[f"{label} questions"]})' for label in COLUMNS]
    lines.append(f'| {row["configuration"]} | {row["arm"]} | {row["repetition"]} | ' + ' | '.join(cells) + ' |')
  return lines
