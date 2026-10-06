import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.common import Claim, check_benchmark
from benchmarks.erb import CLAIMS_KEY, build, convert, unique_document_ids


def _question(question_id: str, question_type: str, expected_doc_ids: list[str], facts: list[str]) -> dict[str, Any]:
  return {
    'question_id': question_id,
    'question_type': question_type,
    'source_types': ['slack'] if expected_doc_ids else [],
    'question': f'Question {question_id}?',
    'expected_doc_ids': expected_doc_ids,
    'gold_answer': f'Answer {question_id}.',
    'answer_facts': facts,
  }


QUESTIONS = [
  _question('qst_0001', 'basic', ['dsid_a'], ['Fact one.', 'Fact two.']),
  _question('qst_0002', 'semantic', ['dsid_b'], ['Fact three.']),
  _question('qst_0003', 'completeness', ['dsid_a', 'dsid_c'], ['Fact four.']),
  _question('qst_0004', 'high_level', [], ['Fact five.']),
  _question('qst_0005', 'conflicting_info', ['dsid_x', 'dsid_x'], ['The answer must mention both.']),
  _question('qst_0006', 'info_not_found', [], ['The answer must say it is unknown.']),
]
# dsid_x labels two documents, as four ids do in the real knowledge base.
DOCUMENT_IDS = ['dsid_a', 'dsid_b', 'dsid_c', 'dsid_x', 'dsid_d', 'dsid_x', 'dsid_e']


class UniqueDocumentIdsTest(unittest.TestCase):
  def test_later_rows_sharing_an_id_get_a_numbered_suffix(self) -> None:
    self.assertEqual(
      unique_document_ids(['a', 'b', 'a', 'c', 'a']),
      ['a', 'b', 'a__2', 'c', 'a__3'],
    )


class ConvertTest(unittest.TestCase):
  def test_keeps_answerable_questions_in_order(self) -> None:
    benchmark = convert(QUESTIONS, DOCUMENT_IDS)

    self.assertEqual([record['sample_key'] for record in benchmark.records], [f'qst_000{i}' for i in range(1, 6)])
    self.assertEqual(benchmark.samples[0].input_prompt, 'Question qst_0001?')
    self.assertEqual(benchmark.samples[0].ground_truth_output, 'Answer qst_0001.')
    self.assertEqual(benchmark.samples[2].document_ids, ['dsid_a', 'dsid_c'])
    self.assertIsNone(benchmark.samples[3].document_ids)
    self.assertEqual(benchmark.records[4]['expected_doc_ids'], ['dsid_x', 'dsid_x'])

  def test_an_id_naming_several_documents_makes_each_of_them_gold(self) -> None:
    benchmark = convert(QUESTIONS, DOCUMENT_IDS)

    self.assertEqual(benchmark.samples[4].document_ids, ['dsid_x', 'dsid_x__2'])
    assert benchmark.knowledge_base_ids is not None
    self.assertIn('dsid_x__2', benchmark.knowledge_base_ids)
    self.assertEqual(len(benchmark.knowledge_base_ids), len(DOCUMENT_IDS))

  def test_only_basic_and_semantic_facts_become_gold_claims(self) -> None:
    claims = convert(QUESTIONS, DOCUMENT_IDS).claims[CLAIMS_KEY]

    self.assertEqual(
      claims,
      {
        'Question qst_0001?': [
          Claim(id='qst_0001-f01', text='Fact one.'),
          Claim(id='qst_0001-f02', text='Fact two.'),
        ],
        'Question qst_0002?': [Claim(id='qst_0002-f01', text='Fact three.')],
      },
    )

  def test_passes_the_checks_and_warns_about_questions_without_gold(self) -> None:
    benchmark = convert(QUESTIONS, DOCUMENT_IDS)
    report = check_benchmark(benchmark, expected_samples=5)

    self.assertTrue(report.passed, report.errors)
    self.assertEqual(len(report.warnings), 1)
    self.assertIn('qst_0004', report.warnings[0])
    self.assertEqual(benchmark.stats['samples_with_gold_documents'], 4)
    self.assertEqual(benchmark.stats['distinct_gold_documents'], 5)
    self.assertEqual(benchmark.stats['claims'], {CLAIMS_KEY: {'samples': 2, 'claims': 3}})

  def test_unknown_gold_ids_fail_the_checks(self) -> None:
    questions = [_question('qst_0001', 'basic', ['dsid_missing'], ['Fact.'])]

    report = check_benchmark(convert(questions, DOCUMENT_IDS), expected_samples=1)

    self.assertIn('dsid_missing', '\n'.join(report.errors))


class BuildTest(unittest.TestCase):
  def test_reads_the_parquet_files_and_reports_content_statistics(self) -> None:
    contents = ['A' * 10, 'B' * 5_000, 'C' * 20, 'nul\x00inside', '', "['From: a list repr']", '["It\'s a list"]']
    with tempfile.TemporaryDirectory() as directory:
      paths = {'questions': Path(directory) / 'questions.parquet', 'documents': Path(directory) / 'documents.parquet'}
      pq.write_table(pa.Table.from_pylist(QUESTIONS), paths['questions'])
      pq.write_table(
        pa.table({'doc_id': DOCUMENT_IDS, 'source_type': ['slack'] * 7, 'title': [''] * 7, 'content': contents}),
        paths['documents'],
      )

      benchmark = build(paths)

    self.assertEqual(len(benchmark.samples), 5)
    content = benchmark.stats['knowledge_base_content']
    assert isinstance(content, dict)
    self.assertEqual(content['empty_documents'], 1)
    self.assertEqual(content['documents_with_nul'], 1)
    self.assertEqual(content['documents_as_list_repr'], 2)
    # Gold documents: dsid_a (10 chars), dsid_b (5,000), dsid_c (20), dsid_x (11) and dsid_x__2 (21).
    self.assertEqual(
      benchmark.stats['gold_document_chars'],
      {'min': 10, 'max': 5_000, 'p50': 20, 'p90': 5_000, 'p99': 5_000, 'over_4000': 1, 'over_8000': 0, 'over_16000': 0},
    )
    claim_gold = benchmark.stats['claim_gold_document_chars']
    assert isinstance(claim_gold, dict)
    self.assertEqual((claim_gold['min'], claim_gold['max']), (10, 5_000))

  def test_rejects_unexpected_question_columns(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      paths = {'questions': Path(directory) / 'questions.parquet', 'documents': Path(directory) / 'documents.parquet'}
      pq.write_table(pa.table({'question_id': ['qst_0001'], 'question': ['Q?']}), paths['questions'])

      with self.assertRaisesRegex(ValueError, 'Unexpected ERB question columns'):
        build(paths)


if __name__ == '__main__':
  unittest.main()
