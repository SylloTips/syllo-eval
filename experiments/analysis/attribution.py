from collections.abc import Collection, Mapping, Sequence
from functools import partial
from typing import Any
from uuid import UUID

from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import MetricComputation, MetricComputationStatus

from ablation_metrics import GoldClaimsContextualRecall, GoldClaimsContextualRecallSC, RetrievedSetRecall
from analysis.relevance import Decision, decision_rows, decision_table
from analysis.runs import JudgeRun, finished_runs, gold_documents, judge_runs, load_attempts, require_judge_outcome
from benchmarks.erb import CLAIMS_KEY
from config import Configuration
from deepeval_baseline import DeepEvalGoldClaimsContextualRecall

ARMS = ('main', 'sc', 'deepeval')
CR_METRICS = frozenset(
  metric.metric_name
  for metric in (GoldClaimsContextualRecall, GoldClaimsContextualRecallSC, DeepEvalGoldClaimsContextualRecall)
)


def decisions(
  computations: Sequence[MetricComputation],
  sample_ids: Mapping[UUID, UUID],
  retrieved: Mapping[UUID, frozenset[str]],
  gold: Mapping[UUID, frozenset[str]],
  claims: Mapping[UUID, list[str]],
) -> list[Decision]:
  found: list[Decision] = []
  for computation in computations:
    sample_id = sample_ids[computation.evaluation_run_sample_id]
    if computation.metric not in CR_METRICS or computation.status is MetricComputationStatus.SKIPPED:
      continue
    if sample_id not in retrieved:
      raise ValueError(f'unit {computation.id} has no retrieved evidence in the source run')
    supported = bool(gold.get(sample_id, frozenset()) & retrieved[sample_id])
    results = (computation.metadata or {}).get('statement_results')
    if computation.status is MetricComputationStatus.FAILED:
      require_judge_outcome(computation)
      found.extend(Decision(sample_id, claim_id, not supported, supported) for claim_id in claims[sample_id])
    elif results:
      found.extend(Decision(sample_id, r['claim_id'], r['attributable'], supported) for r in results)
    else:
      found.extend(Decision(sample_id, claim_id, False, supported, judged=False) for claim_id in claims[sample_id])
  return found


async def _run_decisions(
  db_manager: DatabaseManager, run: JudgeRun, gold: Mapping[UUID, frozenset[str]]
) -> list[Decision]:
  computations, sample_ids = await load_attempts(db_manager, run.run_ids, CR_METRICS)
  evidence, evidence_sample_ids = await load_attempts(
    db_manager, (run.source_run_id,), {RetrievedSetRecall.metric_name}
  )
  retrieved = {
    evidence_sample_ids[c.evaluation_run_sample_id]: frozenset((c.metadata or {}).get('predicted_ids', ()))
    for c in evidence
    if c.status is MetricComputationStatus.COMPLETED
  }
  async with UnitOfWork(db_manager) as uow:
    rows = await uow.ground_truths.list_by_key(CLAIMS_KEY)
  claims = {row.sample_id: [claim['id'] for claim in row.ground_truth_value['claims']] for row in rows}
  return decisions(computations, sample_ids, retrieved, gold, claims)


async def attribution(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  runs = judge_runs(await finished_runs(db_manager), configurations, arms)
  gold = await gold_documents(db_manager)
  runs = [run for run in runs if run.reported and run.configuration.benchmark == 'erb']
  return await decision_rows(runs, lambda run: _run_decisions(db_manager, run, gold))


table = partial(decision_table, gold='Supported', other='Unsupported', unjudged='Without judge')
