#!/usr/bin/env bash
# Index the knowledge base of one benchmark for the agents' search tool: embed every document, then load the
# embeddings into the vector store. Both steps resume where they stopped, so after an interruption run it again.
#
# Usage: scripts/index-benchmark.sh erb|wixqa
#
# The output also goes to outputs/logs/index-<benchmark>-<UTC time>.log. ERB takes about 12 hours, so on macOS the
# script keeps the machine awake while it runs (on AC power, with the lid open).
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(erb|wixqa)$ ]]; then
  echo "usage: $0 erb|wixqa" >&2
  exit 2
fi
benchmark=$1

cd "$(dirname "$0")/.."
log=outputs/logs/index-$benchmark-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$(dirname "$log")"
export PYTHONUNBUFFERED=1

awake() {
  if command -v caffeinate > /dev/null; then
    caffeinate -is "$@"
  else
    "$@"
  fi
}

{
  echo "Indexing $benchmark; the log is $log"
  awake poetry run syllo-exp index embed --only "$benchmark"
  awake poetry run syllo-exp index load --only "$benchmark"
} 2>&1 | tee "$log"
