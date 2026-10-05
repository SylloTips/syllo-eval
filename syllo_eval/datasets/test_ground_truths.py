import unittest
from uuid import UUID

from syllo_eval.datasets.ground_truths import build_sample_ground_truths
from syllo_eval.datasets.model import DatasetJsonPlanStep, DatasetJsonSample
from syllo_eval.model import ExpectedClaim, GroundTruthKey

SAMPLE_ID = UUID('00000000-0000-0000-0000-000000000123')


class BuildSampleGroundTruthsTest(unittest.TestCase):
  def test_full_sample_builds_one_row_per_key(self) -> None:
    sample = DatasetJsonSample(
      input_prompt='Find the contract',
      ground_truth_output=' Contract found ',
      snippet_ids=['sn_1', 'sn_2'],
      document_ids=['doc_1'],
      plan=[
        DatasetJsonPlanStep(operation='retrieve_documents', instruction='Search by contract id.'),
        DatasetJsonPlanStep(operation='answer', instruction='Answer with the contract details.'),
      ],
    )

    ground_truths = build_sample_ground_truths(SAMPLE_ID, sample)

    self.assertEqual(len(ground_truths), 4)
    by_key = {ground_truth.key: ground_truth for ground_truth in ground_truths}
    self.assertEqual(
      set(by_key),
      {
        GroundTruthKey.RELEVANT_DOCUMENT_IDS.value,
        GroundTruthKey.RELEVANT_SNIPPET_IDS.value,
        GroundTruthKey.EXPECTED_OUTPUT.value,
        GroundTruthKey.EXPECTED_PLAN.value,
      },
    )
    self.assertEqual(
      by_key[GroundTruthKey.RELEVANT_DOCUMENT_IDS.value].ground_truth_value,
      {'relevant_ids': ['doc_1']},
    )
    self.assertEqual(
      by_key[GroundTruthKey.RELEVANT_SNIPPET_IDS.value].ground_truth_value,
      {'relevant_ids': ['sn_1', 'sn_2']},
    )
    self.assertEqual(
      by_key[GroundTruthKey.EXPECTED_OUTPUT.value].ground_truth_value,
      {'expected_output': 'Contract found'},
    )
    self.assertEqual(
      by_key[GroundTruthKey.EXPECTED_PLAN.value].ground_truth_value,
      {
        'expected_plan': [
          {'operation': 'retrieve_documents', 'instruction': 'Search by contract id.'},
          {'operation': 'answer', 'instruction': 'Answer with the contract details.'},
        ]
      },
    )
    for ground_truth in ground_truths:
      self.assertEqual(ground_truth.sample_id, SAMPLE_ID)

  def test_minimal_sample_builds_only_relevant_id_rows(self) -> None:
    sample = DatasetJsonSample(input_prompt='Count contracts')

    ground_truths = build_sample_ground_truths(SAMPLE_ID, sample)

    self.assertEqual(
      {ground_truth.key for ground_truth in ground_truths},
      {GroundTruthKey.RELEVANT_DOCUMENT_IDS.value, GroundTruthKey.RELEVANT_SNIPPET_IDS.value},
    )
    for ground_truth in ground_truths:
      self.assertEqual(ground_truth.ground_truth_value, {'relevant_ids': []})

  def test_claims_become_one_expected_claims_row(self) -> None:
    claims = [ExpectedClaim(id='c1', text='Dana approved it.'), ExpectedClaim(id='c2', text='In May.')]

    ground_truths = build_sample_ground_truths(SAMPLE_ID, DatasetJsonSample(input_prompt='Question', claims=claims))

    by_key = {ground_truth.key: ground_truth for ground_truth in ground_truths}
    self.assertEqual(
      by_key[GroundTruthKey.EXPECTED_CLAIMS.value].ground_truth_value,
      {'claims': [{'id': 'c1', 'text': 'Dana approved it.'}, {'id': 'c2', 'text': 'In May.'}]},
    )

  def test_repeated_claim_ids_are_rejected(self) -> None:
    with self.assertRaises(ValueError):
      DatasetJsonSample.model_validate(
        {'input_prompt': 'Question', 'claims': [{'id': 'c1', 'text': 'A.'}, {'id': 'c1', 'text': 'B.'}]}
      )

  def test_blank_expected_output_and_empty_plan_are_skipped(self) -> None:
    sample = DatasetJsonSample(input_prompt='Question', ground_truth_output='   ', plan=[])

    ground_truths = build_sample_ground_truths(SAMPLE_ID, sample)

    keys = {ground_truth.key for ground_truth in ground_truths}
    self.assertNotIn(GroundTruthKey.EXPECTED_OUTPUT.value, keys)
    self.assertNotIn(GroundTruthKey.EXPECTED_PLAN.value, keys)


if __name__ == '__main__':
  unittest.main()
