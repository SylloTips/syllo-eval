import tempfile
import unittest
from pathlib import Path

from pydantic import JsonValue

from syllo_eval.datasets.model import DatasetJsonPayload, DatasetJsonSample

from benchmarks import orbitals_claims
from benchmarks.common import Claim


def _extraction(*claims: tuple[str, str]) -> orbitals_claims.Extraction:
  return orbitals_claims.Extraction.model_validate(
    {'extractions': {'claims': [{'subtype': s, 'content': c, 'evidences': []} for s, c in claims]}, 'model': 'm'}
  )


class _BatchExtractor:
  def __init__(self, *results: list[orbitals_claims.Extraction] | Exception) -> None:
    self.batches: list[list[orbitals_claims.Conversation]] = []
    self._results = list(results)

  async def __call__(self, conversations: list[orbitals_claims.Conversation]) -> list[orbitals_claims.Extraction]:
    self.batches.append(conversations)
    result = self._results.pop(0)
    if isinstance(result, Exception):
      raise result
    return result


PAYLOAD = DatasetJsonPayload(
  samples=[DatasetJsonSample(input_prompt=f'Q{i}?', ground_truth_output=f'A{i}.') for i in range(3)]
)
RECORDS: list[dict[str, JsonValue]] = [{'sample_key': f'wixqa/{i:03d}'} for i in range(3)]


class ExtractTest(unittest.IsolatedAsyncioTestCase):
  async def test_batches_drop_unverifiable_claims_and_a_rerun_extracts_only_what_is_missing(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / orbitals_claims.CLAIMS_FILE
      first = _BatchExtractor(
        [_extraction(('Factoid', 'A0 is 1.'), ('Unverifiable', 'A0 is nice.')), _extraction()], RuntimeError('busy')
      )
      with self.assertRaises(RuntimeError):
        await orbitals_claims.extract(PAYLOAD, RECORDS, path, first, batch_size=2)
      second = _BatchExtractor([_extraction(('Capability', 'A2 can.'))])
      counts = await orbitals_claims.extract(PAYLOAD, RECORDS, path, second, batch_size=2)

      claims = orbitals_claims.read(Path(directory))

    self.assertEqual(first.batches[0][0], [{'role': 'user', 'content': 'Q0?'}, {'role': 'assistant', 'content': 'A0.'}])
    self.assertEqual([len(batch) for batch in first.batches + second.batches], [2, 1, 1])
    self.assertEqual(second.batches[0][0][0]['content'], 'Q2?')
    self.assertEqual(counts, {'samples': 3, 'claims': 2, 'excluded_unverifiable': 1})
    self.assertEqual(
      claims,
      {
        orbitals_claims.CLAIMS_KEY: {
          'Q0?': [Claim(id='wixqa/000-c01', text='A0 is 1.')],
          'Q2?': [Claim(id='wixqa/002-c01', text='A2 can.')],
        }
      },
    )

  async def test_a_batch_answered_with_the_wrong_number_of_extractions_is_refused(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / orbitals_claims.CLAIMS_FILE
      with self.assertRaisesRegex(ValueError, '1 extractions for 3 conversations'):
        await orbitals_claims.extract(PAYLOAD, RECORDS, path, _BatchExtractor([_extraction()]), batch_size=3)
      self.assertFalse(path.exists())


if __name__ == '__main__':
  unittest.main()
