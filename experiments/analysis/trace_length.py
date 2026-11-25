from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from syllo_eval.evaluation.judge.failures import JudgeFailureKind
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork

from ablation_metrics import AnswerCorrectnessWT, PlanCorrectnessWT
from analysis import answers, plans
from analysis.retest import agreement
from analysis.runs import JudgeRun, finished_runs, judge_runs, load_attempts, rounded
from config import Configuration

ARMS = ('main', 'wt')
PARTS = ('short', 'medium', 'long')


async def trace_lengths(db_manager: DatabaseManager, runs: Sequence[JudgeRun], metric: str) -> dict[str, int | None]:
  lengths: dict[str, int | None] = {}
  for run in runs:
    async with UnitOfWork(db_manager) as uow:
      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.source_run_id)
    traces = {run_sample.sample_id: run_sample.trace_id for run_sample in run_samples}
    computations, sample_ids = await load_attempts(db_manager, run.run_ids, {metric})
    for computation in computations:
      metadata = computation.metadata or {}
      trace = traces[sample_ids[computation.evaluation_run_sample_id]]
      if 'trace_render' not in metadata or trace is None:
        continue
      usage = metadata.get('judge_usage')
      if usage is None and metadata.get('failure') != JudgeFailureKind.CONTEXT_OVERFLOW:
        raise ValueError(f'{metric} on trace {trace} has no input token count')
      tokens = None if usage is None else usage['input_tokens']
      if lengths.setdefault(trace, tokens) != tokens:
        raise ValueError(f'trace {trace} counts {lengths[trace]} and {tokens} input tokens')
  return lengths


def terciles(lengths: Mapping[str, int | None]) -> dict[str, str]:
  if len(lengths) < len(PARTS):
    raise ValueError(f'{len(lengths)} traces cannot be split into {len(PARTS)} parts')
  ordered = sorted(lengths, key=lambda key: (lengths[key] is None, lengths[key] or 0, key))
  return {key: PARTS[3 * index // len(ordered)] for index, key in enumerate(ordered)}


def _part(benchmark: str, part: str, lengths: Mapping[str, int | None], parts: Mapping[str, str]) -> dict[str, Any]:
  tokens = [lengths[key] for key, found in parts.items() if found == part]
  counted = [count for count in tokens if count is not None]
  return {
    'benchmark': benchmark,
    'part': part,
    'tokens_min': min(counted),
    'tokens_max': None if None in tokens else max(counted),
    'items': len(tokens),
  }


async def _answer_rows(
  db_manager: DatabaseManager,
  configurations: Sequence[Configuration],
  arms: Collection[str],
  annotations_dir: Path,
  lengths: Mapping[str, int | None],
) -> list[dict[str, Any]]:
  humans, flagged, ties, key = answers.read_annotations(annotations_dir)
  finals = answers.final_grades(humans, flagged, ties)
  if missing := sorted(item for item, (_, _, trace, _) in key.items() if trace not in lengths):
    raise ValueError(f'{len(missing)} annotated answers have no whole-trace length, such as {missing[0]}')
  item_lengths = {item: lengths[trace] for item, (_, _, trace, _) in key.items()}
  parts = terciles(item_lengths)
  scores = await answers.item_scores(db_manager, configurations, arms, key)
  rows = []
  for part in PARTS:
    for arm, repetition in sorted(scores, key=lambda found: (ARMS.index(found[0]), found[1])):
      scored = {item: score for item, score in scores[(arm, repetition)].items() if parts[item] == part}
      rho, n = agreement('ρ', scored, finals)
      rows.append(
        {
          **_part('erb', part, item_lengths, parts),
          'arm': arm,
          'repetition': repetition,
          'scored': n,
          'positives': sum(finals[item] for item in scored),
          'statistic': 'AC ρ',
          'value': rho,
          'matching_auroc': None,
        }
      )
  return rows


async def _plan_rows(
  db_manager: DatabaseManager,
  configurations: Sequence[Configuration],
  arms: Collection[str],
  rewards_dir: Path,
  lengths: Mapping[str, int | None],
) -> list[dict[str, Any]]:
  found, eligible = await plans.trajectories(db_manager, configurations, arms, rewards_dir, plans.ARMS)
  if missing := sorted(eligible - lengths.keys()):
    raise ValueError(f'{len(missing)} τ²-bench trajectories have no whole-trace length, such as {missing[0]}')
  cohort = {trace: lengths[trace] for trace in eligible}
  parts = terciles(cohort)
  groups: defaultdict[tuple[str, str, int], list[plans.Trajectory]] = defaultdict(list)
  for run, scored in found:
    for trace, trajectory in scored:
      groups[(parts[trace], run.arm, run.repetition)].append(trajectory)
  selected = sorted({(run.arm, run.repetition) for run, _ in found}, key=lambda key: (ARMS.index(key[0]), key[1]))
  rows = []
  for part in PARTS:
    for arm, repetition in selected:
      summary = plans.summary(groups[(part, arm, repetition)])
      rows.append(
        {
          **_part('tau2', part, cohort, parts),
          'arm': arm,
          'repetition': repetition,
          'scored': summary['trajectories'],
          'positives': summary['rewarded'],
          'statistic': 'PC AUROC',
          'value': summary['pc_auroc'],
          'matching_auroc': summary['matching_auroc'],
        }
      )
  return rows


async def trace_length(
  db_manager: DatabaseManager,
  configurations: Sequence[Configuration],
  arms: Collection[str],
  annotations_dir: Path,
  rewards_dir: Path,
) -> list[dict[str, Any]]:
  wt = judge_runs(await finished_runs(db_manager), configurations, ('wt',))
  erb = await trace_lengths(
    db_manager, [run for run in wt if run.configuration.benchmark == 'erb'], AnswerCorrectnessWT.metric_name
  )
  tau2 = await trace_lengths(
    db_manager, [run for run in wt if run.configuration.benchmark == 'tau2'], PlanCorrectnessWT.metric_name
  )
  return [
    *await _answer_rows(db_manager, configurations, arms, annotations_dir, erb),
    *await _plan_rows(db_manager, configurations, arms, rewards_dir, tau2),
  ]


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines = [
    '| Benchmark | Part | Tokens | Items | Arm | Rep | Scored | Positives | Statistic | Value | AUROC matching |',
    '|---|---|--:|--:|---|--:|--:|--:|---|--:|--:|',
  ]
  for row in rows:
    tokens = f'{row["tokens_min"]:,}' + ('+' if row['tokens_max'] is None else f'–{row["tokens_max"]:,}')
    lines.append(
      f'| {row["benchmark"]} | {row["part"]} | {tokens} | {row["items"]} | {row["arm"]} | {row["repetition"]} | '
      f'{row["scored"]} | {row["positives"]} | {row["statistic"]} | {rounded(row["value"])} | '
      f'{rounded(row["matching_auroc"])} |'
    )
  return lines
