#!/usr/bin/env bash
# Run Dify's self-hosted stack beside the campaign environment, whose Phoenix receives Dify's traces. Start the campaign
# environment first: Dify's api and worker join its network.
#
# Usage: scripts/dify.sh <docker compose arguments>, e.g. scripts/dify.sh up -d, scripts/dify.sh down
#
# The first run downloads Dify's docker/ folder at the pinned release into data/dify/, checking its SHA-256, and creates
# its .env from .env.example.
set -euo pipefail

version=1.17.1
sha256=ac5df165d788770d09268091fd14690000f96e192b93290b7bb4c51709244ec0

cd "$(dirname "$0")/.."
dir=data/dify
if [[ ! -f $dir/docker-compose.yaml ]]; then
  archive=$(mktemp)
  trap 'rm -f "$archive"' EXIT
  curl -fsSL "https://github.com/langgenius/dify/archive/refs/tags/$version.tar.gz" -o "$archive"
  if ! echo "$sha256  $archive" | shasum -a 256 -c --status; then
    echo "The Dify $version archive does not match its pinned SHA-256" >&2
    exit 1
  fi
  mkdir -p "$dir"
  tar -xzf "$archive" -C "$dir" --strip-components=2 "dify-$version/docker"
  cp "$dir/.env.example" "$dir/.env"
  echo "$version" > "$dir/VERSION"
elif [[ $(cat "$dir/VERSION") != "$version" ]]; then
  echo "$dir holds Dify $(cat "$dir/VERSION"), not $version: stop it and delete the folder to upgrade" >&2
  exit 1
fi

exec docker compose --project-name dify --project-directory "$dir" \
  -f "$dir/docker-compose.yaml" -f dify/compose.override.yaml "$@"
