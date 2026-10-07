"""EnterpriseRAG-Bench (ERB): 480 answerable questions over a 511,962-document enterprise knowledge base.

The paper keeps every question but the 20 ``info_not_found`` ones (Sec. 5.1). Basic and Semantic questions have a single
gold document each, so their answer facts serve as gold claims (Sec. 5.2). Other question types also carry facts, but
for constrained and conflicting questions many of them are grading instructions ("The answer must ..."), not claims.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import JsonValue

from syllo_eval.datasets.model import DatasetJsonSample

from benchmarks.common import (
  Claim,
  ConvertedBenchmark,
  KnowledgeDocument,
  length_summary,
  longest_gold_lengths,
  percentiles,
)

UNANSWERABLE_TYPE = 'info_not_found'
GOLD_CLAIM_TYPES = ('basic', 'semantic')
CLAIMS_KEY = 'expected_claims_gold'
QUESTION_COLUMNS = (
  'question_id',
  'question_type',
  'source_types',
  'question',
  'expected_doc_ids',
  'gold_answer',
  'answer_facts',
)


def unique_document_ids(raw_ids: Sequence[str]) -> list[str]:
  """Knowledge-base ids made unique: the first row keeps its id, later rows sharing it get ``__2``, ``__3``, ...

  Four ERB ids each label two different documents, an original and its near-duplicate rewrite. The search tool and
  every gold list must agree on ids, so this is the one place that assigns them.
  """
  occurrences: Counter[str] = Counter()
  unique_ids = []
  for raw_id in raw_ids:
    occurrences[raw_id] += 1
    count = occurrences[raw_id]
    unique_ids.append(raw_id if count == 1 else f'{raw_id}__{count}')
  return unique_ids


def convert(questions: Sequence[Mapping[str, Any]], raw_document_ids: Sequence[str]) -> ConvertedBenchmark:
  """Convert the rows of the questions file, given the raw ``doc_id`` column of the knowledge base in file order."""
  unique_ids = unique_document_ids(raw_document_ids)
  ids_by_raw_id: dict[str, list[str]] = {}
  for raw_id, unique_id in zip(raw_document_ids, unique_ids):
    ids_by_raw_id.setdefault(raw_id, []).append(unique_id)

  samples: list[DatasetJsonSample] = []
  records: list[dict[str, JsonValue]] = []
  claims: dict[str, list[Claim]] = {}
  for row in questions:
    if row['question_type'] == UNANSWERABLE_TYPE:
      continue
    # An id that names several documents makes each of them gold; unknown ids are kept so that the check reports them.
    gold_ids = [unique_id for raw_id in row['expected_doc_ids'] for unique_id in ids_by_raw_id.get(raw_id, [raw_id])]
    samples.append(
      DatasetJsonSample(
        input_prompt=row['question'],
        ground_truth_output=row['gold_answer'],
        document_ids=list(dict.fromkeys(gold_ids)),
      )
    )
    records.append(
      {
        'sample_key': row['question_id'],
        'question_type': row['question_type'],
        'source_types': list(row['source_types']),
        'expected_doc_ids': list(row['expected_doc_ids']),
        'answer_facts': list(row['answer_facts']),
      }
    )
    if row['question_type'] in GOLD_CLAIM_TYPES:
      claims[row['question']] = [
        Claim(id=f'{row["question_id"]}-f{index:02d}', text=fact) for index, fact in enumerate(row['answer_facts'], 1)
      ]

  gold_ids_used = [doc for sample in samples for doc in sample.document_ids]
  stats: dict[str, JsonValue] = {
    'question_types': dict(Counter(str(record['question_type']) for record in records)),
    'samples_with_gold_documents': sum(1 for sample in samples if sample.document_ids),
    'gold_references': len(gold_ids_used),
    'distinct_gold_documents': len(set(gold_ids_used)),
    'claims': {CLAIMS_KEY: {'samples': len(claims), 'claims': sum(len(facts) for facts in claims.values())}},
    'knowledge_base': {
      'documents': len(raw_document_ids),
      'distinct_raw_ids': len(ids_by_raw_id),
      'renamed_ids': list[JsonValue](sorted(new for new, raw in zip(unique_ids, raw_document_ids) if new != raw)),
    },
  }
  return ConvertedBenchmark(
    samples=samples,
    records=records,
    claims={CLAIMS_KEY: claims},
    knowledge_base_ids=frozenset(unique_ids),
    stats=stats,
  )


def build(paths: Mapping[str, Path]) -> ConvertedBenchmark:
  """Convert the pinned files, adding the content statistics a per-document length cap would be decided on."""
  table = pq.read_table(paths['questions'])
  if tuple(table.column_names) != QUESTION_COLUMNS:
    raise ValueError(f'Unexpected ERB question columns: {table.column_names}')
  documents = pq.read_table(paths['documents'], columns=['doc_id', 'content'])
  raw_ids = documents.column('doc_id').to_pylist()
  benchmark = convert(table.to_pylist(), raw_ids)

  content = documents.column('content')
  lengths = pc.utf8_length(content).to_pylist()
  length_by_id = dict(zip(unique_document_ids(raw_ids), lengths))
  claim_prompts = set(benchmark.claims[CLAIMS_KEY])
  gold_ids = {doc for sample in benchmark.samples for doc in sample.document_ids}
  claim_gold_ids = {
    doc for sample in benchmark.samples if sample.input_prompt in claim_prompts for doc in sample.document_ids
  }
  benchmark.stats['knowledge_base_content'] = {
    'chars': percentiles(lengths),
    'empty_documents': sum(1 for length in lengths if length == 0),
    # PostgreSQL JSONB rejects NUL, so the search tool must remove it before any document reaches a trace.
    'documents_with_nul': pc.sum(pc.match_substring(content, '\x00')).as_py() or 0,
    # Mostly Gmail: the upstream export wrote list-valued bodies as a Python list repr, quoted either way.
    'documents_as_list_repr': pc.sum(pc.or_(pc.starts_with(content, "['"), pc.starts_with(content, '["'))).as_py() or 0,
  }
  # Per distinct gold document, then per question (its longest gold document), which is what a cap costs in scores.
  benchmark.stats['gold_document_chars'] = length_summary(
    [length_by_id[doc] for doc in gold_ids if doc in length_by_id]
  )
  benchmark.stats['longest_gold_document_chars_per_sample'] = length_summary(
    longest_gold_lengths(benchmark.samples, length_by_id)
  )
  # Basic and Semantic questions have one gold document each, and no two share it.
  benchmark.stats['claim_gold_document_chars'] = length_summary(
    [length_by_id[doc] for doc in claim_gold_ids if doc in length_by_id]
  )
  return benchmark


def knowledge_base(paths: Mapping[str, Path], batch_size: int = 4_096) -> Iterator[KnowledgeDocument]:
  """The knowledge-base documents in file order, under the ids the gold lists use.

  The rows are read in batches, because the content column holds 2.5 GB of text.
  """
  documents = pq.ParquetFile(paths['documents'])
  document_ids = unique_document_ids(documents.read(columns=['doc_id']).column('doc_id').to_pylist())
  batches = documents.iter_batches(batch_size, columns=['doc_id', 'source_type', 'title', 'content'])
  rows = (row for batch in batches for row in batch.to_pylist())
  for document_id, row in zip(document_ids, rows, strict=True):
    yield KnowledgeDocument(
      document_id=document_id,
      source_document_id=row['doc_id'],
      title=row['title'],
      text=row['content'].replace('\x00', ''),
      metadata={'source_type': row['source_type']},
    )
