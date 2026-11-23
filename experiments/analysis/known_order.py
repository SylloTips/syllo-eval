from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from uuid import UUID

from syllo_eval.infrastructure import DatabaseManager

from analysis.rankings import METRICS, kendall_tau_b
from analysis.runs import finished_runs, gold_documents, judge_runs, load_attempts, rounded
from analysis.scores import SOURCE_LABELS, question_scores, run_scores, summary
from config import Configuration

ARMS = ('main', 'sc', 'deepeval')


def order_rows(
  runs: Mapping[tuple[str, str, int], Mapping[float, Mapping[str, Mapping[UUID, float]]]],
  fractions: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for (configuration, arm, repetition), by_fraction in runs.items():
    summaries = {fraction: summary(scores) for fraction, scores in by_fraction.items()}
    for metric in METRICS:
      means = {
        fraction: summaries[fraction][metric] if fraction in summaries else None
        for fraction in fractions[configuration]
      }
      if all(mean is None for mean in means.values()):
        continue
      first, last = means[fractions[configuration][0]], means[fractions[configuration][-1]]
      rows.append(
        {
          'configuration': configuration,
          'arm': arm,
          'repetition': repetition,
          'metric': metric,
          'means': means,
          'questions': {
            fraction: summaries[fraction][f'{metric} questions'] if fraction in summaries else 0 for fraction in means
          },
          'tau_b': None
          if None in means.values()
          else kendall_tau_b([mean for mean in means.values() if mean is not None], [-fraction for fraction in means]),
          'drop': (last - first) / first if first and last is not None else None,
        }
      )
  return rows


async def known_order(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str]
) -> list[dict[str, Any]]:
  bases = {_key(c): c.id for c in configurations if not c.degradation}
  fractions: defaultdict[str, list[float]] = defaultdict(lambda: [0.0])
  for c in configurations:
    if c.degradation:
      fractions[bases[_key(c)]].append(c.degradation)
  found = [
    (bases[_key(run.configuration)], run) for run in judge_runs(await finished_runs(db_manager), configurations, arms)
  ]
  degraded = {(base, run.arm, run.repetition) for base, run in found if run.configuration.degradation}
  gold = await gold_documents(db_manager)
  runs: defaultdict[tuple[str, str, int], dict[float, Mapping[str, Mapping[UUID, float]]]] = defaultdict(dict)
  for base, run in found:
    if (base, run.arm, run.repetition) in degraded:
      source, sample_ids = await load_attempts(db_manager, (run.source_run_id,), SOURCE_LABELS)
      runs[(base, run.arm, run.repetition)][run.configuration.degradation or 0.0] = {
        **await run_scores(db_manager, run, gold),
        **question_scores(source, sample_ids, SOURCE_LABELS, gold),
      }
  return order_rows(runs, {base: sorted(values) for base, values in fractions.items()})


def _key(configuration: Configuration) -> tuple[str, str, str]:
  return configuration.benchmark, configuration.agent, configuration.model


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines: list[str] = []
  header: tuple[float, ...] = ()
  for row in rows:
    if tuple(row['means']) != header:
      header = tuple(row['means'])
      lines.extend(
        [
          *([''] if lines else []),
          '| Configuration | Arm | Rep | Metric | '
          + ' | '.join(f'f = {fraction:g}' for fraction in header)
          + f' | τ-b | Drop at f = {header[-1]:g} |',
          '|---|---|--:|---|' + '--:|' * len(header) + '--:|--:|',
        ]
      )
    cells = ['–' if row['means'][f] is None else f'{rounded(row["means"][f])} ({row["questions"][f]})' for f in header]
    lines.append(
      f'| {row["configuration"]} | {row["arm"]} | {row["repetition"]} | {row["metric"]} | '
      + ' | '.join(cells)
      + f' | {rounded(row["tau_b"])} | {"–" if row["drop"] is None else f"{row["drop"]:.1%}"} |'
    )
  return lines
