#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
plugins=data/dify/volumes/plugin_daemon/cwd/langgenius

python3 - "$plugins" <<'EOF'
import sys
from pathlib import Path

PATCHES = {
  'agent-0.0.50@*/strategies/ReAct.py': [
    ('import json\n', 'import json\nimport re\n'),
    (
      '        if (\n'
      '            "Observation" not in stop\n'
      '            and model.provider not in ignore_observation_providers\n'
      '        ):\n',
      '        refuses_stop = re.match(r"gpt-(?:[6-9]|\\d{2,})", model.model) is not None\n'
      '        if (\n'
      '            "Observation" not in stop\n'
      '            and model.provider not in ignore_observation_providers\n'
      '            and not refuses_stop\n'
      '        ):\n',
    ),
    (
      '            for react_chunk in react_chunks:\n'
      '                if isinstance(react_chunk, AgentScratchpadUnit.Action):\n',
      '            for react_chunk in react_chunks:\n'
      '                if refuses_stop and scratchpad.action:\n'
      '                    continue\n'
      '                if isinstance(react_chunk, AgentScratchpadUnit.Action):\n',
    ),
    (
      '                data={"thought": scratchpad.thought, **action_dict},\n',
      '                data={\n'
      '                    "thought": scratchpad.thought,\n'
      '                    **action_dict,\n'
      '                    **(\n'
      '                        {\n'
      '                            "usage": {\n'
      '                                "prompt_tokens": usage_dict["usage"].prompt_tokens,\n'
      '                                "completion_tokens": usage_dict["usage"].completion_tokens,\n'
      '                            }\n'
      '                        }\n'
      '                        if usage_dict["usage"] and usage_dict["usage"].total_tokens\n'
      '                        else {}\n'
      '                    ),\n'
      '                },\n',
    ),
  ],
  'azure_ai_studio-0.0.20@*/models/llm/llm.py': [
    (
      '            response = client.complete(**payload)\n'
      '            if stream:\n'
      '                return self._handle_stream_response(response, model, prompt_messages)\n',
      '            response = client.complete(**{**payload, "stream": False})\n',
    ),
  ],
}

root = Path(sys.argv[1])
for pattern, edits in PATCHES.items():
  paths = list(root.glob(pattern))
  if len(paths) != 1:
    sys.exit(f'Expected one {pattern} under {root}, found {len(paths)}')
  path = paths[0]
  original = source = path.read_text(encoding='utf-8')
  for old, new in edits:
    if new in source:
      continue
    if source.count(old) != 1:
      sys.exit(f'{path}: the code to patch is not as expected: {old.strip().splitlines()[0]!r}')
    source = source.replace(old, new)
  if source == original:
    print(f'{path}: already patched')
    continue
  path.write_text(source, encoding='utf-8')
  print(f'{path}: patched')
EOF

scripts/dify.sh restart plugin_daemon
