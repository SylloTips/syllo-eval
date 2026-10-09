#!/usr/bin/env bash
# Create the tau2 runtime: tau2-bench at the benchmark's pinned commit and the tracing packages, at the versions of
# tau2_runtime/requirements.txt, in experiments/.venv-tau2. tau2 needs its own environment, because its dependencies
# conflict with syllo-eval's.
#
#   scripts/setup-tau2.sh                       # with python3.12
#   PYTHON=python3.13 scripts/setup-tau2.sh     # tau2 supports Python 3.12 and 3.13
set -euo pipefail

cd "$(dirname "$0")/.."
python="${PYTHON:-python3.12}"
venv=.venv-tau2

"$python" -m venv "$venv"
"$venv/bin/python" -m pip install --quiet --disable-pip-version-check --requirement tau2_runtime/requirements.txt
"$venv/bin/python" -m pip check --disable-pip-version-check
echo "tau2 runtime ready: $venv/bin/python"
