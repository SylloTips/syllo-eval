import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from manifest import Manifest, StepStatus


class ManifestTest(unittest.TestCase):
  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.manifest = Manifest(Path(self._directory.name) / 'nested' / 'manifest.jsonl')

  def tearDown(self) -> None:
    self._directory.cleanup()

  def test_missing_manifest_has_no_steps(self) -> None:
    self.assertEqual(self.manifest.records(), [])
    self.assertFalse(self.manifest.is_completed('collect:any'))

  def test_latest_record_of_a_step_is_its_state(self) -> None:
    run_id = uuid4()
    self.manifest.append('collect:a', StepStatus.STARTED)
    self.manifest.append('collect:b', StepStatus.FAILED, details={'error': 'agent unavailable'})
    self.manifest.append('collect:a', StepStatus.COMPLETED, run_ids=(run_id,))

    latest = self.manifest.latest()
    self.assertEqual(len(self.manifest.records()), 3)
    self.assertEqual(latest['collect:a'].run_ids, (run_id,))
    self.assertTrue(self.manifest.is_completed('collect:a'))
    self.assertFalse(self.manifest.is_completed('collect:b'))
    self.assertEqual(latest['collect:b'].details, {'error': 'agent unavailable'})

  def test_a_new_attempt_reopens_a_completed_step(self) -> None:
    self.manifest.append('judge:pass-1', StepStatus.COMPLETED)
    self.manifest.append('judge:pass-1', StepStatus.STARTED)

    self.assertFalse(self.manifest.is_completed('judge:pass-1'))


if __name__ == '__main__':
  unittest.main()
