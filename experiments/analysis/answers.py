import csv
import hashlib
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork

from analysis.retest import agreement
from analysis.runs import finished_runs, judge_runs, load_attempts, rounded
from analysis.scores import LABELS, question_scores
from config import Configuration

ARMS = ('main', 'wt', 'deepeval')
ANNOTATORS = ('A', 'B', 'C', 'D')
AC = {metric: label for metric, label in LABELS.items() if label == 'AC'}
Key = dict[str, tuple[str, str, str]]


def _one(directory: Path, pattern: str) -> dict[str, Any]:
  paths = sorted(directory.glob(pattern))
  if len(paths) != 1:
    raise ValueError(f'{directory} needs exactly one {pattern}, found {len(paths)}')
  return json.loads(paths[0].read_text(encoding='utf-8'))


def read_annotations(directory: Path) -> tuple[dict[str, dict[str, int]], set[str], dict[str, Any], Key]:
  with (directory / 'coordinator-key.csv').open(encoding='utf-8', newline='') as handle:
    key = {
      row['item_id']: (row['configuration'], row['collection_run_id'], row['trace_id'])
      for row in csv.DictReader(handle)
    }
  exports = {letter: _one(directory, f'erb-task1-{letter}-*.json') for letter in ANNOTATORS}
  ties = _one(directory, 'erb-task1-ties-*.json')
  packet = (directory / 'task1-items.json').read_bytes()
  packets = {export['packet'] for export in (*exports.values(), ties)}
  if packets != {hashlib.sha256(packet).hexdigest()}:
    raise ValueError(f'the annotation files do not all come from task1-items.json: {sorted(packets)}')
  shown = {item['item_id']: item['agent_answer'] for item in json.loads(packet)}
  with (directory / 'answers.jsonl').open(encoding='utf-8') as handle:
    traced = {record['trace_id']: record['agent_answer'] for record in map(json.loads, handle)}
  wrong = [item for item, (_, _, trace) in key.items() if traced.get(trace) != shown.get(item)]
  if shown.keys() != key.keys() or wrong:
    raise ValueError(f'the coordinator key does not match the packet: {len(wrong)} of its traces answer other items')
  for letter, export in exports.items():
    if export['grades'].keys() != key.keys():
      raise ValueError(f'annotator {letter} graded other items than the coordinator key lists')
  humans = {
    letter: {item: grade['grade'] for item, grade in export['grades'].items() if grade['grade'] is not None}
    for letter, export in exports.items()
  }
  flagged = {item for export in exports.values() for item, grade in export['grades'].items() if grade['flagged']}
  return humans, flagged, {item: final['grade'] for item, final in ties['finals'].items()}, key


def final_grades(
  humans: Mapping[str, Mapping[str, int]], flagged: Collection[str], ties: Mapping[str, Any]
) -> dict[str, int]:
  finals: dict[str, int] = {}
  settled: set[str] = set()
  for item in sorted(set().union(*humans.values(), flagged)):
    votes = [grades[item] for grades in humans.values() if item in grades]
    if any(vote not in (0, 1) for vote in votes):
      raise ValueError(f'{item} has a grade other than 0 and 1')
    if item in flagged or 2 * sum(votes) == len(votes):
      if ties.get(item) not in (0, 1):
        raise ValueError(f'{item} is tied or flagged and needs a tie-break grade of 0 or 1')
      finals[item] = ties[item]
      settled.add(item)
    else:
      finals[item] = int(2 * sum(votes) > len(votes))
  if extra := sorted(ties.keys() - settled):
    raise ValueError(f'tie-break grades for items with a majority: {extra}')
  return finals


def fleiss_kappa(ratings: Sequence[Sequence[int]]) -> float | None:
  if not ratings:
    return None
  raters = len(ratings[0])
  share = mean(map(sum, ratings)) / raters
  observed = mean((sum(r) ** 2 + (raters - sum(r)) ** 2 - raters) / (raters * (raters - 1)) for r in ratings)
  chance = share**2 + (1 - share) ** 2
  return None if chance == 1 else (observed - chance) / (1 - chance)


def _row(
  rater: str, repetition: int | None, reference: str, statistic: str, value: float | None, items: int | None
) -> dict[str, Any]:
  return {
    'rater': rater,
    'repetition': repetition,
    'reference': reference,
    'statistic': statistic,
    'items': items,
    'value': value,
  }


def _mean(rater: str, repetition: int | None, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  values = [row['value'] for row in rows]
  return _row(rater, repetition, 'annotators (mean)', 'ρ', None if None in values else mean(values), None)


def human_rows(
  humans: Mapping[str, Mapping[str, int]], flagged: Collection[str], ties: Mapping[str, Any], finals: Mapping[str, int]
) -> list[dict[str, Any]]:
  pairs = [_row(a, None, b, 'ρ', *agreement('ρ', humans[a], humans[b])) for a, b in combinations(ANNOTATORS, 2)]
  complete = [item for item in finals if all(item in humans[letter] for letter in ANNOTATORS)]
  ratings = [[humans[letter][item] for letter in ANNOTATORS] for item in complete]
  splits = Counter(max(sum(r), len(r) - sum(r)) for r in ratings)
  return [
    *pairs,
    _mean('annotators', None, pairs),
    _row('annotators', None, 'Fleiss', 'κ', fleiss_kappa(ratings), len(complete)),
    *(_row('annotators', None, f'{v}–{4 - v}', 'items', None, splits[v]) for v in (4, 3, 2)),
    _row('annotators', None, 'flagged', 'items', None, len(flagged)),
    _row('annotators', None, 'tie-break', 'items', None, len(ties)),
    *(
      _row(letter, None, 'graded correct', 'share', mean(humans[letter].values()), len(humans[letter]))
      for letter in ANNOTATORS
    ),
    _row('final', None, 'graded correct', 'share', mean(finals.values()), len(finals)),
  ]


def rater_rows(
  arm: str,
  repetition: int,
  scores: Mapping[str, float],
  humans: Mapping[str, Mapping[str, int]],
  finals: Mapping[str, int],
) -> list[dict[str, Any]]:
  per = [_row(arm, repetition, letter, 'ρ', *agreement('ρ', scores, grades)) for letter, grades in humans.items()]
  return [_row(arm, repetition, 'final', 'ρ', *agreement('ρ', scores, finals)), *per, _mean(arm, repetition, per)]


async def item_scores(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str], key: Key
) -> dict[tuple[str, int], dict[str, float]]:
  annotated = {(configuration, run_id) for configuration, run_id, _ in key.values()}
  runs = [
    run
    for run in await finished_runs(db_manager)
    if ((run.metadata or {}).get('configuration'), str(run.source_run_id)) in annotated
  ]
  scores: dict[tuple[str, int], dict[str, float]] = {}
  for run in judge_runs(runs, configurations, arms):
    configuration = run.configuration.id
    async with UnitOfWork(db_manager) as uow:
      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.source_run_id)
    samples = {run_sample.trace_id: run_sample.sample_id for run_sample in run_samples}
    computations, sample_ids = await load_attempts(db_manager, run.run_ids, AC)
    ac = question_scores(computations, sample_ids, AC, {}).get('AC', {})
    for item, (item_configuration, _, trace_id) in key.items():
      if item_configuration != configuration:
        continue
      if trace_id not in samples:
        raise ValueError(f'run {run.source_run_id} has no trace {trace_id}')
      if samples[trace_id] in ac:
        scores.setdefault((run.arm, run.repetition), {})[item] = ac[samples[trace_id]]
  return scores


async def answers(
  db_manager: DatabaseManager, configurations: Sequence[Configuration], arms: Collection[str], annotations_dir: Path
) -> list[dict[str, Any]]:
  humans, flagged, ties, key = read_annotations(annotations_dir)
  finals = final_grades(humans, flagged, ties)
  rows = human_rows(humans, flagged, ties, finals)
  scores = await item_scores(db_manager, configurations, arms, key)
  for arm, repetition in sorted(scores, key=lambda found: (ARMS.index(found[0]), found[1])):
    rows.extend(rater_rows(arm, repetition, scores[(arm, repetition)], humans, finals))
  return rows


def table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
  lines = ['| Rater | Rep | Reference | Statistic | Items | Value |', '|---|--:|---|---|--:|--:|']
  lines.extend(
    f'| {row["rater"]} | {"–" if row["repetition"] is None else row["repetition"]} | {row["reference"]} | '
    f'{row["statistic"]} | {"–" if row["items"] is None else row["items"]} | {rounded(row["value"])} |'
    for row in rows
  )
  return lines
