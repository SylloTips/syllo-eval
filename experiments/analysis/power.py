from collections.abc import Collection, Mapping, Sequence
from itertools import combinations
from typing import Any
from uuid import UUID

import numpy as np

from syllo_eval.infrastructure import DatabaseManager

from analysis.rankings import METRICS, configuration_scores
from analysis.runs import rounded
from config import Configuration

ARMS = ('main', 'sc', 'deepeval')
RESAMPLES = 10_000
ALPHA = 0.05
SEED = 0
TOLERANCE = 1e-9


def bootstrap_p_value(differences: np.ndarray, rng: np.random.Generator) -> float:
  observed = differences.mean()
  means = differences[rng.integers(0, len(differences), (RESAMPLES, len(differences)))].mean(axis=1)
  return float((np.sum(np.abs(means - observed) >= abs(observed) - TOLERANCE) + 1) / (RESAMPLES + 1))


def holm(p_values: Sequence[float]) -> list[float]:
  adjusted = [0.0] * len(p_values)
  running = 0.0
  for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
    running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
    adjusted[index] = running
  return adjusted


def compare_pairs(
  scored: Mapping[str, Mapping[Any, float]], labels: Sequence[str]
) -> tuple[list[dict[str, Any]], int | None]:
  pairs: list[dict[str, Any]] = []
  for a, b in combinations(labels, 2):
    questions = sorted(scored[a].keys() & scored[b].keys())
    differences = np.array([scored[a][q] - scored[b][q] for q in questions])
    pairs.append(
      {
        'pair': f'{a} vs {b}',
        'questions': len(questions),
        'difference': float(differences.mean()) if questions else None,
        'p': bootstrap_p_value(differences, np.random.default_rng(SEED)) if len(questions) > 1 else None,
      }
    )
  complete = bool(pairs) and all(pair['p'] is not None for pair in pairs)
  if complete:
    for pair, p_holm in zip(pairs, holm([pair['p'] for pair in pairs])):
      pair['p_holm'] = p_holm
  return pairs, sum(pair['p_holm'] < ALPHA for pair in pairs) if complete else None


def power_rows(
  groups: Mapping[tuple[str, str, int], Mapping[str, Mapping[str, Mapping[UUID, float]]]],
  expected: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for (benchmark, arm, repetition), by_configuration in groups.items():
    labels = expected[benchmark]
    for metric in METRICS:
      scored = {label: by_configuration.get(label, {}).get(metric) or {} for label in labels}
      if not any(scored.values()):
        continue
      pairs, significant = compare_pairs(scored, labels)
      rows.append(
        {
          'benchmark': benchmark,
          'arm': arm,
          'repetition': repetition,
          'metric': metric,
          'significant': significant,
          'power': None if significant is None else significant / len(pairs),
          'pairs': pairs,
        }
      )
  return rows


async def power(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  return power_rows(*await configuration_scores(db_manager, configurations, arms))


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines = [
    '| Benchmark | Arm | Rep | Metric | Questions | Significant pairs | Power |',
    '|---|---|--:|---|---|--:|--:|',
  ]
  for row in rows:
    counts = sorted({pair['questions'] for pair in row['pairs']})
    questions = '–' if not counts else str(counts[0]) if counts[0] == counts[-1] else f'{counts[0]}–{counts[-1]}'
    significant = '–' if row['significant'] is None else f'{row["significant"]}/{len(row["pairs"])}'
    lines.append(
      f'| {row["benchmark"]} | {row["arm"]} | {row["repetition"]} | {row["metric"]} | {questions} | {significant} | '
      f'{rounded(row["power"])} |'
    )
  return lines
