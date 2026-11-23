from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations
from math import sqrt
from typing import Any
from uuid import UUID

from syllo_eval.infrastructure import DatabaseManager

from analysis.runs import finished_runs, gold_documents, judge_runs, reported, rounded
from analysis.scores import run_scores, summary, with_r_on_cr
from config import Configuration

ARMS = ('main', 'sc', 'deepeval')
METRICS = ('AC', 'CP', 'CR', 'R', 'R on CR', 'NDCG@10')
REFERENCES = {'CP': ('NDCG@10',), 'CR': ('R', 'R on CR')}


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
  pairs = list(combinations(range(len(x)), 2))
  sx = [(x[i] > x[j]) - (x[i] < x[j]) for i, j in pairs]
  sy = [(y[i] > y[j]) - (y[i] < y[j]) for i, j in pairs]
  untied = sum(map(abs, sx)) * sum(map(abs, sy))
  return sum(a * b for a, b in zip(sx, sy)) / sqrt(untied) if untied else None


def ranking_rows(
  groups: Mapping[tuple[str, str, int], Mapping[str, Mapping[str, Mapping[UUID, float]]]],
  expected: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for (benchmark, arm, repetition), by_configuration in groups.items():
    summaries = {label: summary(scores) for label, scores in by_configuration.items()}
    scored = {
      metric: {label: summaries[label][metric] if label in summaries else None for label in expected[benchmark]}
      for metric in METRICS
    }
    means = {metric: values for metric, values in scored.items() if any(value is not None for value in values.values())}
    complete = {metric for metric, values in means.items() if None not in values.values()}
    for metric, values in means.items():
      present = [value for value in values.values() if value is not None]
      rows.append(
        {
          'benchmark': benchmark,
          'arm': arm,
          'repetition': repetition,
          'metric': metric,
          'means': values,
          'questions': {
            label: summaries[label][f'{metric} questions'] if label in summaries else 0 for label in values
          },
          'ranks': {
            label: None if value is None else 1 + sum(other > value for other in present)
            for label, value in values.items()
          },
          'tau_b': {
            reference: kendall_tau_b(present, [value for value in means[reference].values() if value is not None])
            if {metric, reference} <= complete
            else None
            for reference in REFERENCES.get(metric, ())
            if reference in means
          },
        }
      )
  return rows


async def rankings(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  return ranking_rows(*await configuration_scores(db_manager, configurations, arms))


async def configuration_scores(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> tuple[dict[tuple[str, str, int], dict[str, Mapping[str, Mapping[UUID, float]]]], dict[str, list[str]]]:
  gold = await gold_documents(db_manager)
  groups: defaultdict[tuple[str, str, int], dict[str, Mapping[str, Mapping[UUID, float]]]] = defaultdict(dict)
  for run in judge_runs(await finished_runs(db_manager), configurations, arms):
    if run.reported:
      groups[(run.configuration.benchmark, run.arm, run.repetition)][_label(run.configuration)] = with_r_on_cr(
        await run_scores(db_manager, run, gold)
      )
  expected: defaultdict[str, list[str]] = defaultdict(list)
  for configuration in configurations:
    if reported(configuration):
      expected[configuration.benchmark].append(_label(configuration))
  return groups, expected


def _label(configuration: Configuration) -> str:
  return f'{configuration.agent}/{configuration.model}'


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines: list[str] = []
  header: tuple[str, ...] = ()
  for row in rows:
    if tuple(row['means']) != header:
      header = tuple(row['means'])
      lines.extend(
        [
          *([''] if lines else []),
          '| Benchmark | Arm | Rep | Metric | ' + ' | '.join(header) + ' | τ-b |',
          '|---|---|--:|---|' + '--:|' * len(header) + '---|',
        ]
      )
    cells = [
      '–'
      if row['means'][label] is None
      else f'{rounded(row["means"][label])} ({row["questions"][label]}) #{row["ranks"][label]}'
      for label in header
    ]
    tau = ', '.join(f'{rounded(value)} vs {reference}' for reference, value in row['tau_b'].items()) or '–'
    lines.append(
      f'| {row["benchmark"]} | {row["arm"]} | {row["repetition"]} | {row["metric"]} | '
      + ' | '.join(cells)
      + f' | {tau} |'
    )
  return lines
