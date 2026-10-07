import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmarks.common import KnowledgeDocument, check_benchmark
from benchmarks.wixqa import build, convert, knowledge_base

EXPERT_WRITTEN = [
  {
    'question': 'How do I connect a domain?',
    'answer': '1. Open **Domains**.  \n2. Connect it.\n',
    'article_ids': ['k1'],
  },
  {'question': 'Can I refund an order?\n\n', 'answer': 'Yes.', 'article_ids': ['k2', 'k3']},
]
SIMULATED = [{'question': 'How to publish?', 'answer': 'Click Publish. ', 'article_ids': ['k2']}]
KNOWLEDGE_BASE = [
  {
    'id': 'k1',
    'url': 'https://support.example/1',
    'contents': 'Domains\nBody',
    'title': 'Domains',
    'article_type': 'article',
  },
  {'id': 'k2', 'url': 'https://support.example/2', 'contents': 'R' * 5_000, 'title': 'R', 'article_type': 'article'},
  {
    'id': 'k3',
    'url': 'https://support.example/3',
    'contents': 'Feature',
    'title': 'F',
    'article_type': 'feature_request',
  },
]


def _convert() -> Any:
  return convert({'wixqa_expertwritten': EXPERT_WRITTEN, 'wixqa_simulated': SIMULATED}, KNOWLEDGE_BASE)


class ConvertTest(unittest.TestCase):
  def test_samples_follow_config_then_row_order_with_stable_keys(self) -> None:
    benchmark = _convert()

    self.assertEqual(
      [record['sample_key'] for record in benchmark.records],
      ['wixqa_expertwritten/000', 'wixqa_expertwritten/001', 'wixqa_simulated/000'],
    )
    self.assertEqual(benchmark.samples[1].document_ids, ['k2', 'k3'])
    self.assertEqual(benchmark.records[2]['config'], 'wixqa_simulated')

  def test_strips_only_the_ends_and_records_the_raw_question_when_it_changed(self) -> None:
    benchmark = _convert()

    self.assertEqual(benchmark.samples[1].input_prompt, 'Can I refund an order?')
    self.assertEqual(benchmark.records[1]['question_raw'], 'Can I refund an order?\n\n')
    self.assertNotIn('question_raw', benchmark.records[0])
    # Markdown hard breaks inside the answer survive.
    self.assertEqual(benchmark.samples[0].ground_truth_output, '1. Open **Domains**.  \n2. Connect it.')
    self.assertEqual(benchmark.records[1]['question_sha256'], hashlib.sha256(b'Can I refund an order?').hexdigest())

  def test_passes_the_checks_against_the_knowledge_base(self) -> None:
    benchmark = _convert()
    report = check_benchmark(benchmark, expected_samples=3)

    self.assertTrue(report.passed, report.errors)
    self.assertEqual(report.warnings, [])
    self.assertEqual(benchmark.knowledge_base_ids, frozenset({'k1', 'k2', 'k3'}))
    self.assertEqual(benchmark.stats['gold_references'], 4)
    self.assertEqual(benchmark.stats['distinct_gold_documents'], 3)
    gold_chars = benchmark.stats['gold_document_chars']
    assert isinstance(gold_chars, dict)
    self.assertEqual((gold_chars['max'], gold_chars['over_4000']), (5_000, 1))
    # The long article k2 is gold for two questions: one document, two questions a 4k cap would cut.
    per_sample = benchmark.stats['longest_gold_document_chars_per_sample']
    assert isinstance(per_sample, dict) and isinstance(per_sample['all'], dict)
    self.assertEqual(per_sample['all']['over_4000'], 2)
    simulated = per_sample['wixqa_simulated']
    assert isinstance(simulated, dict)
    self.assertEqual(simulated['over_4000'], 1)

  def test_gold_articles_missing_from_the_knowledge_base_fail_the_checks(self) -> None:
    queries = {
      'wixqa_expertwritten': [{'question': 'Q?', 'answer': 'A.', 'article_ids': ['missing']}],
      'wixqa_simulated': [],
    }

    report = check_benchmark(convert(queries, KNOWLEDGE_BASE), expected_samples=1)

    self.assertIn('missing', '\n'.join(report.errors))


class BuildTest(unittest.TestCase):
  def test_reads_the_jsonl_files(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      paths = {}
      for role, rows in (
        ('expertwritten', EXPERT_WRITTEN),
        ('simulated', SIMULATED),
        ('knowledge_base', KNOWLEDGE_BASE),
      ):
        paths[role] = Path(directory) / f'{role}.jsonl'
        paths[role].write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

      benchmark = build(paths)

    self.assertEqual(len(benchmark.samples), 3)
    self.assertEqual(benchmark.samples[2].input_prompt, 'How to publish?')


class KnowledgeBaseTest(unittest.TestCase):
  def test_reads_the_articles_in_order_with_their_url_and_type(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'kb.jsonl'
      path.write_text(''.join(json.dumps(row) + '\n' for row in KNOWLEDGE_BASE), encoding='utf-8')

      documents = list(knowledge_base({'knowledge_base': path}))

    self.assertEqual([document.document_id for document in documents], ['k1', 'k2', 'k3'])
    self.assertEqual(
      documents[0],
      KnowledgeDocument(
        document_id='k1',
        source_document_id='k1',
        title='Domains',
        text='Domains\nBody',
        metadata={'url': 'https://support.example/1', 'article_type': 'article'},
      ),
    )
    self.assertEqual(documents[2].metadata['article_type'], 'feature_request')


if __name__ == '__main__':
  unittest.main()
