from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from statistics import correlation, mean
from typing import Any

from syllo_eval.infrastructure import DatabaseManager

from analysis import attribution, relevance
from analysis.relevance import cohen_kappa
from analysis.runs import finished_runs, gold_documents, judge_runs, rounded
from analysis.scores import run_scores
from config import Configuration

ARMS = ('main',)
PAIRS = ((1, 2), (1, 3), (2, 3))
DECISIONS = ('Relevance', 'Attribution')
SCORES = ('CP', 'CR', 'AC', 'PC')


def agreement(measure: str, first: Mapping[Any, Any], second: Mapping[Any, Any]) -> tuple[float | None, int]:
  keys = first.keys() & second.keys()
  if measure in DECISIONS:
    return cohen_kappa([(first[key], second[key]) for key in keys]), len(keys)
  return correlation([first[key] for key in keys], [second[key] for key in keys], method='ranked'), len(keys)


def retest_rows(values: Mapping[tuple[str, str, int], Mapping[Any, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for benchmark, measure in dict.fromkeys(key[:2] for key in values):
    pairs = [
      (f'{a}-{b}', *agreement(measure, values[(benchmark, measure, a)], values[(benchmark, measure, b)]))
      for a, b in PAIRS
      if (benchmark, measure, a) in values and (benchmark, measure, b) in values
    ]
    statistic = 'κ' if measure in DECISIONS else 'ρ'
    rows.extend(
      {'benchmark': benchmark, 'measure': measure, 'statistic': statistic, 'pair': pair, 'items': n, 'agreement': value}
      for pair, value, n in pairs
    )
    if len(pairs) == len(PAIRS) and all(value is not None for _, value, _ in pairs):
      rows.append(
        {
          'benchmark': benchmark,
          'measure': measure,
          'statistic': statistic,
          'pair': 'mean',
          'items': min(n for _, _, n in pairs),
          'agreement': mean(value for _, value, _ in pairs if value is not None),
        }
      )
  return rows


async def retest(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  gold = await gold_documents(db_manager)
  values: defaultdict[tuple[str, str, int], dict[Any, Any]] = defaultdict(dict)
  for run in judge_runs(await finished_runs(db_manager), configurations, arms):
    benchmark, configuration = run.configuration.benchmark, run.configuration.id
    if benchmark != 'tau2' and not run.reported:
      continue
    if benchmark != 'tau2':
      for decision in await relevance.run_decisions(db_manager, run, gold):
        values[(benchmark, 'Relevance', run.repetition)][(configuration, decision.sample_id, decision.item_id)] = (
          decision.accepted
        )
    if benchmark == 'erb':
      for decision in await attribution.run_decisions(db_manager, run, gold):
        values[(benchmark, 'Attribution', run.repetition)][(configuration, decision.sample_id, decision.item_id)] = (
          decision.accepted
        )
    for label, by_sample in (await run_scores(db_manager, run, gold)).items():
      if label in SCORES:
        values[(benchmark, label, run.repetition)].update(
          {(configuration, sample_id): score for sample_id, score in by_sample.items()}
        )
  return retest_rows(values)


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines = ['| Benchmark | Measure | Statistic | Pair | Items | Agreement |', '|---|---|---|---|--:|--:|']
  lines.extend(
    f'| {row["benchmark"]} | {row["measure"]} | {row["statistic"]} | {row["pair"]} | {row["items"]} | '
    f'{rounded(row["agreement"])} |'
    for row in rows
  )
  return lines
