"""Command-line entry point: ``poetry run syllo-exp <command>``. Run it from the ``experiments/`` folder."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from config import CONFIG_DIR, load_config
from manifest import Manifest

DEFAULT_MANIFEST = Path(__file__).resolve().parent / 'outputs' / 'manifest.jsonl'


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog='syllo-exp', description='Reproduce the experiments of the paper.')
  commands = parser.add_subparsers(dest='command', required=True)

  configurations = commands.add_parser('configurations', help='Validate the configs and list agent configurations.')
  configurations.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')

  steps = commands.add_parser('steps', help='Show the current state of every recorded experiment step.')
  steps.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.command == 'configurations':
    config = load_config(args.config_dir)
    for configuration in config.configurations:
      model = config.models.agents[configuration.model]
      print(
        f'{configuration.id:<34} agent={configuration.agent:<15} version={configuration.version_tag:<16} {model.label}'
      )
    print(f'{len(config.configurations)} configurations; judge: {config.models.judge.label}')
  elif args.command == 'steps':
    for step, record in sorted(Manifest(args.manifest).latest().items()):
      runs = ', '.join(str(run_id) for run_id in record.run_ids) or '-'
      print(f'{step:<48} {record.status.value:<10} {record.recorded_at:%Y-%m-%d %H:%M} runs: {runs}')
  return 0
