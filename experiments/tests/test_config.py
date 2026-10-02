import unittest
from typing import Any

from pydantic import ValidationError

from config import Configuration, ExperimentConfig, load_config


def _models() -> dict[str, Any]:
  return {
    'agents': {'model-a': {'provider': 'test', 'model': 'model-a-1', 'label': 'Model A'}},
    'customer_simulator': {'provider': 'test', 'model': 'customer-1', 'label': 'Customer'},
    'judge': {
      'provider': 'gemini',
      'model': 'judge-1',
      'label': 'Judge',
      'temperature': 0.0,
      'timeout_seconds': 300,
      'max_retries': 5,
      'max_concurrent_requests': 4,
    },
  }


class ConfigurationShapeTest(unittest.TestCase):
  def test_version_tag_combines_model_with_degradation_or_trial(self) -> None:
    plain = Configuration(id='kb/plain', benchmark='erb', agent='react', model='model-a')
    degraded = Configuration(id='kb/degraded', benchmark='erb', agent='react', model='model-a', degradation=0.25)
    trial = Configuration(id='tasks/trial', benchmark='tau2', agent='tau2-llm-agent', model='model-a', trial=3)

    self.assertEqual(plain.version_tag, 'model-a')
    self.assertEqual(degraded.version_tag, 'model-a-f0.25')
    self.assertEqual(trial.version_tag, 'model-a-trial3')

  def test_tau2_needs_its_agent_and_a_trial(self) -> None:
    invalid: list[dict[str, Any]] = [
      {'benchmark': 'tau2', 'agent': 'react', 'trial': 1},
      {'benchmark': 'erb', 'agent': 'tau2-llm-agent'},
      {'benchmark': 'tau2', 'agent': 'tau2-llm-agent'},
      {'benchmark': 'wixqa', 'agent': 'odr', 'trial': 1},
      {'benchmark': 'tau2', 'agent': 'tau2-llm-agent', 'trial': 1, 'degradation': 0.5},
    ]
    for fields in invalid:
      with self.subTest(fields=fields), self.assertRaises(ValidationError):
        Configuration(id='invalid', model='model-a', **fields)


class ExperimentConfigTest(unittest.TestCase):
  def test_rejects_undefined_models_duplicate_ids_and_clashing_agent_versions(self) -> None:
    invalid = {
      'undefined model': [{'id': 'a', 'benchmark': 'erb', 'agent': 'react', 'model': 'missing'}],
      'duplicate id': [
        {'id': 'a', 'benchmark': 'erb', 'agent': 'react', 'model': 'model-a'},
        {'id': 'a', 'benchmark': 'wixqa', 'agent': 'react', 'model': 'model-a'},
      ],
      'same agent version': [
        {'id': 'a', 'benchmark': 'erb', 'agent': 'react', 'model': 'model-a'},
        {'id': 'b', 'benchmark': 'erb', 'agent': 'react', 'model': 'model-a'},
      ],
    }
    for case, configurations in invalid.items():
      with self.subTest(case=case), self.assertRaises(ValidationError):
        ExperimentConfig.model_validate({'models': _models(), 'configurations': configurations})

  def test_committed_configs_describe_the_paper_setup(self) -> None:
    config = load_config()
    by_benchmark: dict[str, list[Configuration]] = {}
    for configuration in config.configurations:
      by_benchmark.setdefault(configuration.benchmark, []).append(configuration)

    # Section 5.1: 3 agents x 2 models on ERB and WixQA, plus two degraded ReAct/Sonnet runs on ERB;
    # 2 models x 4 trials on tau2.
    self.assertEqual(len(by_benchmark['wixqa']), 6)
    self.assertEqual(len(by_benchmark['erb']), 8)
    self.assertEqual(sorted(c.degradation for c in by_benchmark['erb'] if c.degradation), [0.25, 0.5])
    self.assertEqual(len(by_benchmark['tau2']), 8)
    self.assertEqual(config.configuration('erb/react/sonnet/f0.5').version_tag, 'sonnet-f0.5')
    with self.assertRaises(KeyError):
      config.configuration('missing')


if __name__ == '__main__':
  unittest.main()
