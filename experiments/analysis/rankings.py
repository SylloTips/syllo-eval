from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations
from math import sqrt
from statistics import mean
from typing import Any
from uuid import UUID

from syllo_eval.infrastructure import DatabaseManager

from analysis.runs import finished_runs, gold_documents, judge_runs, rounded
from analysis.scores import run_scores, with_r_on_cr
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
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for (benchmark, arm, repetition), by_configuration in groups.items():
    paired = {
      metric: set.intersection(*(set(scores.get(metric, {})) for scores in by_configuration.values()))
      for metric in METRICS
    }
    means = {
      metric: {label: mean(scores[metric][q] for q in questions) for label, scores in by_configuration.items()}
      for metric, questions in paired.items()
      if questions
    }
    rows.extend(
      {
        'benchmark': benchmark,
        'arm': arm,
        'repetition': repetition,
        'metric': metric,
        'questions': len(paired[metric]),
        'means': values,
        'ranks': {label: 1 + sum(other > value for other in values.values()) for label, value in values.items()},
        'tau_b': {
          reference: kendall_tau_b(list(values.values()), [means[reference][label] for label in values])
          for reference in REFERENCES.get(metric, ())
          if reference in means
        },
      }
      for metric, values in means.items()
    )
  return rows


async def rankings(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  gold = await gold_documents(db_manager)
  groups: defaultdict[tuple[str, str, int], dict[str, Mapping[str, Mapping[UUID, float]]]] = defaultdict(dict)
  for run in judge_runs(await finished_runs(db_manager), configurations, arms):
    if run.reported:
      label = f'{run.configuration.agent}/{run.configuration.model}'
      groups[(run.configuration.benchmark, run.arm, run.repetition)][label] = with_r_on_cr(
        await run_scores(db_manager, run, gold)
      )
  return ranking_rows(groups)


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines: list[str] = []
  header: tuple[str, ...] = ()
  for row in rows:
    if tuple(row['means']) != header:
      header = tuple(row['means'])
      lines.extend(
        [
          *([''] if lines else []),
          '| Benchmark | Arm | Rep | Metric | Questions | ' + ' | '.join(header) + ' | τ-b |',
          '|---|---|--:|---|--:|' + '--:|' * len(header) + '---|',
        ]
      )
    cells = [f'{rounded(row["means"][label])} ({row["ranks"][label]})' for label in header]
    tau = ', '.join(f'{rounded(value)} vs {reference}' for reference, value in row['tau_b'].items()) or '–'
    lines.append(
      f'| {row["benchmark"]} | {row["arm"]} | {row["repetition"]} | {row["metric"]} | {row["questions"]} | '
      + ' | '.join(cells)
      + f' | {tau} |'
    )
  return lines
