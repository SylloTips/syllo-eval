import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cli import main
from manifest import Manifest, StepStatus


def _run(argv: list[str]) -> tuple[int, str]:
  output = io.StringIO()
  with redirect_stdout(output):
    status = main(argv)
  return status, output.getvalue()


class CliTest(unittest.TestCase):
  def test_configurations_lists_every_agent_configuration(self) -> None:
    status, output = _run(['configurations'])

    self.assertEqual(status, 0)
    self.assertIn('erb/react/sonnet/f0.25', output)
    self.assertIn('version=deepseek-trial4', output)
    self.assertIn('22 configurations', output)

  def test_steps_shows_the_latest_state_of_each_step(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      manifest_path = Path(directory) / 'manifest.jsonl'
      manifest = Manifest(manifest_path)
      manifest.append('collect:erb/react/sonnet', StepStatus.STARTED)
      manifest.append('collect:erb/react/sonnet', StepStatus.COMPLETED)

      status, output = _run(['steps', '--manifest', str(manifest_path)])

    self.assertEqual(status, 0)
    self.assertEqual(output.count('collect:erb/react/sonnet'), 1)
    self.assertIn('completed', output)


if __name__ == '__main__':
  unittest.main()
