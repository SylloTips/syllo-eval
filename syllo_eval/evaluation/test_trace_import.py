import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter
from syllo_eval.evaluation.trace_import import (
  ImportedTrace,
  TraceImportError,
  load_trace_file,
  match_traces_to_samples,
  normalize_imported_trace,
)
from syllo_eval.model import Sample


def phoenix_trace(trace_id: str, question: Any) -> ImportedTrace:
  return ImportedTrace(
    trace_id=trace_id,
    spans=[
      {
        'name': 'agent',
        'span_kind': 'chain',
        'parent_id': None,
        'start_time': '2026-01-01T00:00:00+00:00',
        'end_time': '2026-01-01T00:00:01+00:00',
        'context.span_id': f'{trace_id}-root',
        'attributes.input.value': question,
      }
    ],
  )


class TestTraceImport(unittest.TestCase):
  def setUp(self) -> None:
    dataset_id = uuid4()
    self.first = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='First question?')
    self.second = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='Second question?')
    self.adapter = PhoenixTraceAdapter()

  def normalize(self, trace_id: str, question: Any) -> Any:
    return normalize_imported_trace(self.adapter, phoenix_trace(trace_id, question))

  def test_traces_bind_to_the_sample_with_the_same_prompt(self) -> None:
    results = [self.normalize('t1', ' First question? '), self.normalize('t2', 'Second question?')]

    self.assertEqual(
      match_traces_to_samples(results, [self.first, self.second]), {self.first.id: 't1', self.second.id: 't2'}
    )

  def test_every_trace_must_bind_to_exactly_one_distinct_sample(self) -> None:
    duplicate = Sample(id=uuid4(), dataset_id=self.first.dataset_id, input_prompt='First question?')
    cases = {
      'no sample': ([self.normalize('t1', 'Unknown question?')], [self.first], 'matches 0 dataset samples'),
      'ambiguous sample': ([self.normalize('t1', 'First question?')], [self.first, duplicate], 'matches 2'),
      'same sample twice': (
        [self.normalize('t1', 'First question?'), self.normalize('t2', 'First question?')],
        [self.first],
        'match the same sample',
      ),
      'no request text': ([self.normalize('t1', {'messages': []})], [self.first], 'no request text'),
    }
    for name, (results, samples, message) in cases.items():
      with self.subTest(name), self.assertRaisesRegex(TraceImportError, message):
        match_traces_to_samples(results, samples)

  def test_invalid_records_and_files_raise_import_errors(self) -> None:
    broken = ImportedTrace(trace_id='t1', spans=[{'name': 'no span id'}])
    with self.assertRaisesRegex(TraceImportError, 'Trace t1 could not be normalized'):
      normalize_imported_trace(self.adapter, broken)

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'trace.json'
      path.write_text('{"trace_id": "t1", "spans": []}')
      with self.assertRaisesRegex(TraceImportError, 'Invalid trace file'):
        load_trace_file(path)
      path.write_text(phoenix_trace('t1', 'First question?').model_dump_json())
      self.assertEqual(load_trace_file(path).trace_id, 't1')


if __name__ == '__main__':
  unittest.main()
