"""WixQA: the 400 ExpertWritten and Simulated customer-support queries over the 6,221-article Wix Help Center snapshot.

The query files have no question id, so a sample is keyed by its config and 0-based row index at the pinned revision.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from syllo_eval.datasets.model import DatasetJsonSample

from benchmarks.common import (
  ConvertedBenchmark,
  KnowledgeDocument,
  length_summary,
  longest_gold_lengths,
  percentiles,
)

QUERY_FILES = {'wixqa_expertwritten': 'expertwritten', 'wixqa_simulated': 'simulated'}


def convert(
  queries_by_config: Mapping[str, Sequence[Mapping[str, Any]]], knowledge_base: Sequence[Mapping[str, Any]]
) -> ConvertedBenchmark:
  """Convert the query rows of each config, in file order, against the knowledge-base rows."""
  samples: list[DatasetJsonSample] = []
  records: list[dict[str, JsonValue]] = []
  for config in QUERY_FILES:
    for index, row in enumerate(queries_by_config[config]):
      # Two ExpertWritten questions end in newlines; stripping changes nothing else.
      prompt = row['question'].strip()
      samples.append(
        DatasetJsonSample(
          input_prompt=prompt, ground_truth_output=row['answer'].strip(), document_ids=list(row['article_ids'])
        )
      )
      record: dict[str, JsonValue] = {
        'sample_key': f'{config}/{index:03d}',
        'config': config,
        'row_index': index,
        'question_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
      }
      if prompt != row['question']:
        record['question_raw'] = row['question']
      records.append(record)

  article_ids = [str(article['id']) for article in knowledge_base]
  length_by_id = {str(article['id']): len(article['contents']) for article in knowledge_base}
  gold_ids = [doc for sample in samples for doc in sample.document_ids or ()]
  stats: dict[str, JsonValue] = {
    'samples_by_config': dict(Counter(str(record['config']) for record in records)),
    'gold_documents_per_sample': {
      str(k): v for k, v in sorted(Counter(len(s.document_ids or ()) for s in samples).items())
    },
    'gold_references': len(gold_ids),
    'distinct_gold_documents': len(set(gold_ids)),
    'knowledge_base': {
      'articles': len(article_ids),
      'distinct_ids': len(set(article_ids)),
      'article_types': dict(Counter(str(article['article_type']) for article in knowledge_base)),
      'content_chars': percentiles(list(length_by_id.values())),
    },
    # Per distinct gold document, then per question (its longest gold document), which is what a cap costs in scores.
    'gold_document_chars': length_summary([length_by_id[doc] for doc in set(gold_ids) if doc in length_by_id]),
    'longest_gold_document_chars_per_sample': {
      'all': length_summary(longest_gold_lengths(samples, length_by_id)),
      **{
        config: length_summary(
          longest_gold_lengths(
            [sample for sample, record in zip(samples, records) if record['config'] == config], length_by_id
          )
        )
        for config in QUERY_FILES
      },
    },
  }
  return ConvertedBenchmark(samples=samples, records=records, knowledge_base_ids=frozenset(article_ids), stats=stats)


def build(paths: Mapping[str, Path]) -> ConvertedBenchmark:
  queries = {config: read_jsonl(paths[role]) for config, role in QUERY_FILES.items()}
  return convert(queries, read_jsonl(paths['knowledge_base']))


def knowledge_base(paths: Mapping[str, Path]) -> Iterator[KnowledgeDocument]:
  """The Help Center articles in file order. An article's ``contents`` starts with its title."""
  for article in read_jsonl(paths['knowledge_base']):
    article_id = str(article['id'])
    yield KnowledgeDocument(
      document_id=article_id,
      source_document_id=article_id,
      title=article['title'],
      text=article['contents'].replace('\x00', ''),
      metadata={'url': article['url'], 'article_type': article['article_type']},
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  with path.open(encoding='utf-8') as file:
    return [json.loads(line) for line in file if line.strip()]
