import unittest
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from syllo_eval.model import (
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  MetricComputation,
  MetricComputationStatus,
  MetricTargetingMode,
)

import json
import tempfile
from pathlib import Path

from analysis import attribution, plans
from analysis.rankings import ranking_rows
from analysis.relevance import Decision, decisions, summary
from analysis.retest import retest_rows
from analysis.runs import judge_runs, latest_attempts, reportable_runs, rounded
from analysis.scores import question_scores
from analysis.scores import summary as score_summary
from config import Configuration

CONFIGURATIONS = [
  Configuration(id='erb/react/gpt', benchmark='erb', agent='react', model='gpt'),
  Configuration(id='erb/odr/gpt', benchmark='erb', agent='odr', model='gpt'),
]
START = datetime(2026, 11, 1, tzinfo=UTC)


SOURCE = uuid4()


def _run(metadata: dict[str, Any] | None, source: UUID = SOURCE, minutes: int = 0) -> EvaluationRun:
  return EvaluationRun(
    id=uuid4(),
    agent_id=uuid4(),
    dataset_id=uuid4(),
    status=EvaluationStatus.COMPLETED,
    start_time=START + timedelta(minutes=minutes),
    source_run_id=source,
    metadata=metadata,
  )


def _run_sample(sample: UUID, status: EvaluationSampleStatus = EvaluationSampleStatus.COMPLETED) -> EvaluationRunSample:
  return EvaluationRunSample(id=uuid4(), evaluation_run_id=uuid4(), sample_id=sample, status=status)


def _computation(
  run_sample: UUID, span: str, verdicts: list[tuple[str, bool]] | None, failure: str | None = None
) -> MetricComputation:
  computation = _unit(run_sample, 'contextual_precision_document_judge_sc', failure=failure)
  computation.span_ids = [span]
  if not failure:
    computation.metadata = {'rank_results': [{'retrieved_id': doc, 'relevant': ok} for doc, ok in verdicts or []]}
  return computation


class JudgeRunsTest(unittest.TestCase):
  def test_selects_the_arms_in_config_order_and_skips_collection_and_pilot_runs(self) -> None:
    odr = _run({'arm': 'main', 'configuration': 'erb/odr/gpt', 'repetition': 1})
    react_sc = _run({'arm': 'sc', 'configuration': 'erb/react/gpt', 'repetition': 1})
    react = _run({'arm': 'main', 'configuration': 'erb/react/gpt', 'repetition': 1, 'step': 'judge:s:main:1'})
    pilot = _run({'arm': 'main', 'configuration': 'erb/react/gpt', 'repetition': 1, 'step': 'judge:s:main:1:pilot'})
    deepeval = _run({'arm': 'deepeval', 'configuration': 'erb/react/gpt', 'repetition': 1})
    collection = _run({'configuration': 'erb/react/gpt'})
    collection.source_run_id = None

    found = judge_runs([odr, react_sc, react, pilot, deepeval, collection], CONFIGURATIONS, {'main', 'sc'})

    self.assertEqual([run.run_ids for run in found], [(react.id,), (react_sc.id,), (odr.id,)])
    self.assertEqual([run.reported for run in found], [True, True, False])

  def test_runs_on_a_pilot_dataset_are_left_out(self) -> None:
    pilot_dataset = uuid4()
    pilot_collection = _run(None)
    pilot_collection.source_run_id, pilot_collection.dataset_id = None, pilot_dataset
    metadata = {'arm': 'deepeval', 'configuration': 'erb/react/gpt', 'repetition': 1}
    on_pilot = _run(metadata, source=pilot_collection.id)
    on_pilot.dataset_id = pilot_dataset
    full = _run(metadata)

    (found,) = judge_runs(
      reportable_runs([pilot_collection, on_pilot, full], {pilot_dataset}), CONFIGURATIONS, {'deepeval'}
    )

    self.assertEqual(found.run_ids, (full.id,))

  def test_retries_join_their_step_in_start_order(self) -> None:
    metadata = {'arm': 'main', 'configuration': 'erb/react/gpt', 'repetition': 2}
    retry = _run({**metadata, 'step': 'judge:s:main:2:retry-1'}, minutes=5)
    first = _run({**metadata, 'step': 'judge:s:main:2'})

    (found,) = judge_runs([retry, first], CONFIGURATIONS, {'main'})

    self.assertEqual((found.run_ids, found.source_run_id), ((first.id, retry.id), SOURCE))

  def test_crgs_runs_are_the_main_arm_s_first_recall_v2_pass(self) -> None:
    main = _run({'arm': 'main', 'configuration': 'erb/react/gpt', 'repetition': 1})
    crgs = _run({'arm': 'crgs', 'configuration': 'erb/react/gpt', 'repetition': 1}, minutes=5)

    (found,) = judge_runs([crgs, main], CONFIGURATIONS, {'main'})

    self.assertEqual((found.arm, found.run_ids), ('main', (main.id, crgs.id)))

  def test_runs_of_one_step_on_different_source_runs_are_an_error(self) -> None:
    metadata = {'arm': 'main', 'configuration': 'erb/react/gpt', 'repetition': 2}
    with self.assertRaisesRegex(ValueError, 'several source runs'):
      judge_runs([_run(metadata), _run(metadata, source=uuid4())], CONFIGURATIONS, {'main'})


class LatestAttemptsTest(unittest.TestCase):
  def test_each_sample_comes_from_the_latest_run_that_completed_it_with_the_metric(self) -> None:
    kept, retried, unfinished = uuid4(), uuid4(), uuid4()
    first = [_run_sample(kept), _run_sample(retried)]
    second = [_run_sample(retried), _run_sample(kept, EvaluationSampleStatus.FAILED)]
    third = [_run_sample(kept), _run_sample(unfinished, EvaluationSampleStatus.FAILED)]
    computations = [
      [_computation(first[0].id, 'a', []), _computation(first[1].id, 'a', [])],
      [_computation(second[0].id, 'b', []), _computation(second[1].id, 'b', [])],
      [_computation(third[1].id, 'c', [])],
    ]
    answer_only = _computation(third[0].id, 'c', [])
    answer_only.metric = 'answer_correctness_judge'
    computations[2].append(answer_only)

    found, sample_ids = latest_attempts(
      list(zip((first, second, third), computations)),
      {'contextual_precision_document_judge_sc', 'answer_correctness_judge'},
    )

    self.assertEqual(
      sorted((sample_ids[c.evaluation_run_sample_id], c.metric[0], c.span_ids[0]) for c in found),
      sorted([(kept, 'c', 'a'), (retried, 'c', 'b'), (kept, 'a', 'c')]),
    )


class RelevanceDecisionsTest(unittest.TestCase):
  def setUp(self) -> None:
    self.sample, self.unlabeled = uuid4(), uuid4()
    self.run_sample, self.unlabeled_run_sample = uuid4(), uuid4()
    self.sample_ids = {self.run_sample: self.sample, self.unlabeled_run_sample: self.unlabeled}
    self.gold = {self.sample: frozenset({'g1', 'g2'}), self.unlabeled: frozenset()}
    self.starts = {'early': START, 'late': START + timedelta(seconds=5), 'failed': START + timedelta(seconds=9)}

  def test_keeps_the_first_occurrence_of_each_document_by_search_order(self) -> None:
    computations = [
      _computation(self.run_sample, 'late', [('g1', False), ('d1', True)]),
      _computation(self.run_sample, 'early', [('d1', False), ('g1', True)]),
      _computation(self.unlabeled_run_sample, 'early', [('d2', True)]),
    ]

    found = decisions(computations, self.sample_ids, self.starts, self.gold, {})

    self.assertEqual(
      sorted(found, key=lambda decision: decision.item_id),
      [Decision(self.sample, 'd1', False, False), Decision(self.sample, 'g1', True, True)],
    )

  def test_a_failed_unit_counts_each_document_of_its_ranking_as_wrong(self) -> None:
    computations = [_computation(self.run_sample, 'failed', None, failure='truncated')]

    found = decisions(computations, self.sample_ids, self.starts, self.gold, {'failed': ['g2', 'd3']})

    self.assertEqual(found, [Decision(self.sample, 'g2', False, True), Decision(self.sample, 'd3', True, False)])

  def test_an_infrastructure_failure_stops_the_analysis(self) -> None:
    computations = [_computation(self.run_sample, 'failed', None, failure='provider')]
    with self.assertRaisesRegex(ValueError, 'score it again'):
      decisions(computations, self.sample_ids, self.starts, self.gold, {})

  def test_summary_shares_and_kappa(self) -> None:
    sample = uuid4()
    rows = [
      Decision(sample, 'g1', True, True),
      Decision(sample, 'g2', False, True),
      Decision(sample, 'd1', True, False),
      Decision(sample, 'd2', False, False),
      Decision(sample, 'd3', False, False),
      Decision(sample, 'd4', False, False),
    ]

    result = summary(rows)

    self.assertEqual((result['questions'], result['gold'], result['other']), (1, 2, 4))
    self.assertEqual((result['gold_accepted'], result['other_accepted']), (0.5, 0.25))
    self.assertAlmostEqual(result['kappa'], 0.25)


def _unit(
  run_sample: UUID,
  metric: str,
  score: float | None = None,
  skip: str | None = None,
  failure: str | None = None,
) -> MetricComputation:
  status = (
    MetricComputationStatus.FAILED
    if failure
    else MetricComputationStatus.SKIPPED
    if skip
    else MetricComputationStatus.COMPLETED
  )
  return MetricComputation(
    id=uuid4(),
    evaluation_run_sample_id=run_sample,
    metric=metric,
    targeting_mode=MetricTargetingMode.SINGLE,
    target_span_type='retrieval',
    score=score,
    status=status,
    error_message=skip,
    metadata={'failure': failure} if failure else None,
  )


class QuestionScoresTest(unittest.TestCase):
  LABELS = {'cp': 'CP', 'cr': 'CR', 'ac': 'AC', 'pc': 'PC', 'ndcg': 'NDCG@10'}

  def _scores(self, units: dict[str, list[tuple[str, dict[str, Any]]]], gold: dict[str, frozenset[str]]) -> dict:
    run_samples = {name: uuid4() for name in units}
    computations = [
      _unit(run_samples[name], metric, **kwargs) for name, group in units.items() for metric, kwargs in group
    ]
    found = question_scores(
      computations,
      {rs: rs for rs in run_samples.values()},
      self.LABELS,
      {run_samples[name]: labels for name, labels in gold.items()},
    )
    names = {rs: name for name, rs in run_samples.items()}
    return {label: {names[sample]: score for sample, score in by_sample.items()} for label, by_sample in found.items()}

  def test_each_question_weighs_the_same_whatever_its_number_of_searches(self) -> None:
    found = self._scores(
      {
        'one': [('cp', {'score': 0.9}), ('ndcg', {'score': 0.6})],
        'many': [('cp', {'score': 0.1})] * 4 + [('ndcg', {'score': 0.2}), ('ndcg', {'score': 0.4})],
      },
      {'one': frozenset({'g'}), 'many': frozenset({'g'})},
    )

    self.assertEqual(found['CP'], {'one': 0.9, 'many': 0.1})
    self.assertAlmostEqual(found['NDCG@10']['many'], 0.3)

  def test_skips_score_zero_or_leave_the_question_out_as_the_methodology_says(self) -> None:
    no_search = 'No spans found for target span type "retrieval".'
    found = self._scores(
      {
        'no_search': [('cp', {'skip': no_search}), ('ndcg', {'skip': no_search}), ('cr', {'score': 0.0})],
        'no_gold': [('cp', {'skip': no_search}), ('ndcg', {'skip': no_search})],
        'unreadable': [
          ('cp', {'score': 0.8}),
          ('cr', {'skip': 'Document retrieval unavailable: unreadable search outputs: x'}),
        ],
        'no_answer': [('ac', {'skip': 'Trace does not provide an answer.'})],
        'no_plan': [('pc', {'skip': 'Missing required ground truth: expected_plan.'})],
        'failed': [
          ('cp', {'score': 1.0}),
          ('cp', {'failure': 'misaligned'}),
          ('ac', {'failure': 'context_overflow'}),
        ],
        'all_failed': [('cr', {'failure': 'timeout'})],
      },
      {name: frozenset({'g'}) for name in ('no_search', 'unreadable', 'failed')},
    )

    self.assertEqual(found['CP'], {'no_search': 0.0, 'no_gold': 0.0, 'failed': 1.0})
    self.assertEqual(found['NDCG@10'], {'no_search': 0.0})
    self.assertEqual(found['CR'], {'no_search': 0.0})
    self.assertEqual(found['AC'], {'no_answer': 0.0})
    self.assertNotIn('PC', found)

  def test_set_recall_on_cr_keeps_only_the_questions_cr_scores(self) -> None:
    scored, other = uuid4(), uuid4()
    result = score_summary({'CR': {scored: 0.5}, 'R': {scored: 1.0, other: 0.0}})

    self.assertEqual((result['R'], result['R questions']), (0.5, 2))
    self.assertEqual((result['R on CR'], result['R on CR questions']), (1.0, 1))

  def test_an_infrastructure_failure_stops_the_analysis(self) -> None:
    with self.assertRaisesRegex(ValueError, 'score it again'):
      self._scores({'q': [('cr', {'failure': 'provider'})]}, {})


class RankingsTest(unittest.TestCase):
  def test_ranks_paired_means_and_compares_them_with_tau_b(self) -> None:
    q1, q2 = uuid4(), uuid4()
    scores = {
      'a': {'CP': {q1: 0.9}, 'NDCG@10': {q1: 0.8}, 'CR': {q1: 0.5}, 'R': {q1: 1.0}},
      'b': {'CP': {q1: 0.5}, 'NDCG@10': {q1: 0.2}, 'CR': {q1: 0.4}, 'R': {q1: 1.0}},
      'c': {'CP': {q1: 0.1, q2: 1.0}, 'NDCG@10': {q1: 0.4}, 'CR': {q1: 0.1}, 'R': {q1: 0.0}},
    }
    rows = {row['metric']: row for row in ranking_rows({('erb', 'main', 1): scores})}

    self.assertEqual((rows['CP']['questions'], rows['CP']['means']['c']), (1, 0.1))
    self.assertAlmostEqual(rows['CP']['tau_b']['NDCG@10'], 1 / 3)
    self.assertEqual(rows['R']['ranks'], {'a': 1, 'b': 1, 'c': 3})
    self.assertAlmostEqual(rows['CR']['tau_b']['R'], 2 / 6**0.5)
    self.assertEqual(rows['NDCG@10']['tau_b'], {})


class RetestTest(unittest.TestCase):
  def test_pairs_the_same_items_across_passes_and_averages_only_complete_triples(self) -> None:
    first = {('erb/react/gpt', SOURCE, claim): accepted for claim, accepted in zip('abcd', (True, False, True, False))}
    third = {**dict(zip(first, (True, False, False, True))), ('erb/react/gpt', SOURCE, 'e'): True}
    values: dict[tuple[str, str, int], dict[Any, Any]] = {
      ('erb', 'Attribution', 1): first,
      ('erb', 'Attribution', 2): first,
      ('erb', 'Attribution', 3): third,
      ('erb', 'CR', 1): {'x': 0.1, 'y': 0.5, 'z': 0.9},
      ('erb', 'CR', 2): {'x': 0.2, 'y': 0.6, 'z': 1.0},
      ('erb', 'AC', 1): {'x': 1.0, 'y': 1.0},
      ('erb', 'AC', 2): {'x': 1.0, 'y': 1.0},
      ('erb', 'AC', 3): {'x': 1.0},
    }

    rows = [
      (row['measure'], row['statistic'], row['pair'], row['items'], rounded(row['agreement']))
      for row in retest_rows(values)
    ]

    self.assertEqual(
      rows,
      [
        ('Attribution', 'κ', '1-2', 4, '1.000'),
        ('Attribution', 'κ', '1-3', 4, '0.000'),
        ('Attribution', 'κ', '2-3', 4, '0.000'),
        ('Attribution', 'κ', 'mean', None, '0.333'),
        ('CR', 'ρ', '1-2', 3, '1.000'),
        ('AC', 'ρ', '1-2', 2, '–'),
        ('AC', 'ρ', '1-3', 1, '–'),
        ('AC', 'ρ', '2-3', 1, '–'),
      ],
    )


class RoundedTest(unittest.TestCase):
  def test_rounds_half_up_like_the_paper(self) -> None:
    self.assertEqual([rounded(257 / 400), rounded(0.0604), rounded(None)], ['0.643', '0.060', '–'])


class FactSupportDecisionsTest(unittest.TestCase):
  def setUp(self) -> None:
    self.retrieved_gold, self.missed_gold, self.no_evidence = uuid4(), uuid4(), uuid4()
    self.run_samples = {sample: uuid4() for sample in (self.retrieved_gold, self.missed_gold, self.no_evidence)}
    self.sample_ids = {run_sample: sample for sample, run_sample in self.run_samples.items()}
    self.gold = {sample: frozenset({f'gold-{sample}'}) for sample in self.run_samples}
    self.retrieved = {
      self.retrieved_gold: frozenset({f'gold-{self.retrieved_gold}', 'd1'}),
      self.missed_gold: frozenset({'d2'}),
      self.no_evidence: frozenset(),
    }

  def _recall(
    self, sample: UUID, verdicts: list[tuple[str, bool]] | None, failure: str | None = None
  ) -> MetricComputation:
    computation = _unit(self.run_samples[sample], 'contextual_recall_gold_claims_siblings_question', failure=failure)
    if verdicts is not None:
      computation.metadata = {
        'statement_results': [{'claim_id': claim, 'attributable': accepted} for claim, accepted in verdicts]
      }
    return computation

  def test_every_fact_is_a_decision_and_facts_without_evidence_are_rejected_without_the_judge(self) -> None:
    computations = [
      self._recall(self.retrieved_gold, [('f1', True), ('f2', False)]),
      self._recall(self.missed_gold, [('f3', True), ('f4', False)]),
      self._recall(self.no_evidence, []),
    ]

    claims = {self.no_evidence: ['f5', 'f6']}

    found = attribution.decisions(computations, self.sample_ids, self.retrieved, self.gold, claims)

    self.assertEqual(
      found,
      [
        Decision(self.retrieved_gold, 'f1', True, True),
        Decision(self.retrieved_gold, 'f2', False, True),
        Decision(self.missed_gold, 'f3', True, False),
        Decision(self.missed_gold, 'f4', False, False),
        Decision(self.no_evidence, 'f5', False, False, judged=False),
        Decision(self.no_evidence, 'f6', False, False, judged=False),
      ],
    )
    self.assertEqual(summary(found)['unjudged'], 2)

  def test_a_failed_unit_counts_each_fact_of_its_ground_truth_as_wrong(self) -> None:
    computations = [
      self._recall(self.retrieved_gold, None, failure='truncated'),
      self._recall(self.missed_gold, None, failure='misaligned'),
    ]
    claims = {self.retrieved_gold: ['f1'], self.missed_gold: ['f2']}

    found = attribution.decisions(computations, self.sample_ids, self.retrieved, self.gold, claims)

    self.assertEqual(
      found, [Decision(self.retrieved_gold, 'f1', False, True), Decision(self.missed_gold, 'f2', True, False)]
    )

  def test_skips_leave_the_question_out_and_other_failures_stop_the_analysis(self) -> None:
    skipped = _unit(
      self.run_samples[self.missed_gold], 'contextual_recall_gold_claims_siblings_question', skip='Missing ground truth'
    )
    self.assertEqual(attribution.decisions([skipped], self.sample_ids, {}, self.gold, {}), [])
    with self.assertRaisesRegex(ValueError, 'score it again'):
      attribution.decisions(
        [self._recall(self.missed_gold, None, failure='provider')], self.sample_ids, self.retrieved, self.gold, {}
      )
    with self.assertRaisesRegex(ValueError, 'no retrieved evidence'):
      attribution.decisions([self._recall(self.missed_gold, [('f', True)])], self.sample_ids, {}, self.gold, {})


class PlansTest(unittest.TestCase):
  def test_matching_is_one_minus_the_normalized_edit_distance(self) -> None:
    find, cancel = plans.call('find', {'id': 1}), plans.call('cancel', {'id': 1, 'why': 'x'})
    self.assertEqual(plans.matching([find, cancel], [find, cancel]), 1.0)
    self.assertEqual(plans.matching([], []), 1.0)
    self.assertEqual(plans.matching([find, plans.call('cancel', {'why': 'x', 'id': 2})], [find, cancel]), 0.5)
    self.assertEqual(plans.matching([find, find, cancel], [cancel]), 1 - 2 / 3)

  def test_auroc_counts_ties_as_half(self) -> None:
    self.assertEqual(plans.auroc([(1.0, True), (0.0, False)]), 1.0)
    self.assertEqual(plans.auroc([(0.7, True), (0.7, False), (1.0, True)]), 0.75)
    self.assertIsNone(plans.auroc([(1.0, True)]))

  def test_deviating_successes_are_rewarded_trajectories_that_fail_the_action_check(self) -> None:
    trajectories = [
      plans.Trajectory(1.0, 0.2, True, False),
      plans.Trajectory(0.7, 0.4, True, False),
      plans.Trajectory(1.0, 1.0, True, True),
      plans.Trajectory(0.0, 0.1, False, False),
    ]

    result = plans.summary(trajectories)

    self.assertEqual((result['deviating'], result['deviating_pc']), (2, 0.85))
    self.assertAlmostEqual(result['deviating_matching'], 0.3)

  def test_rewards_come_from_each_sample_s_latest_attempt(self) -> None:
    sample = uuid4()
    lines = [
      {'sample_id': str(sample), 'attempt': 1, 'reward': 0.0, 'reward_info': {'action_checks': []}},
      {
        'sample_id': str(sample),
        'attempt': 2,
        'reward': 1.0,
        'reward_info': {'action_checks': [{'action_match': True}, {'action_match': False}]},
      },
    ]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'rewards.jsonl'
      path.write_text('\n'.join(json.dumps(line) for line in lines))
      self.assertEqual(plans.read_rewards(path), {sample: (True, False)})
