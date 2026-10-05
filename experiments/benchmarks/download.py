"""Fetch pinned benchmark files: resume interrupted downloads, and keep a file only once its size and SHA-256 match.

Each file is fetched under an exclusive lock (POSIX ``flock``), so two fetches of the same file cannot interleave.
"""

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from benchmarks.common import file_sha256
from config import BenchmarkSource, PinnedFile

_CHUNK_BYTES = 1 << 20
_ATTEMPTS = 5
_BACKOFF_SECONDS = 2.0
# Ask for the stored bytes: a compressed response would make byte ranges count compressed bytes, which some hosts
# (raw.githubusercontent.com) do while the resume offset counts decompressed ones.
_HEADERS = {'Accept-Encoding': 'identity'}


class PinnedFileMismatchError(RuntimeError):
  """Raised when a file's size or SHA-256 differs from its pin."""


class FetchInProgressError(RuntimeError):
  """Raised when another process is already fetching the same file."""


class _RetryableStatusError(httpx.HTTPStatusError):
  """A rate limit, a server error or an unusable range: the download is worth retrying."""


def fetch_pinned_files(source: BenchmarkSource, raw_dir: Path, client: httpx.Client) -> dict[str, Path]:
  """Download every pinned file of ``source`` into ``raw_dir`` (by its repository path), verifying each one."""
  return {
    role: fetch_pinned_file(source.base_url, pinned, raw_dir / pinned.path, client)
    for role, pinned in source.files.items()
  }


def fetch_pinned_file(
  base_url: str,
  pinned: PinnedFile,
  destination: Path,
  client: httpx.Client,
  *,
  attempts: int = _ATTEMPTS,
  backoff_seconds: float = _BACKOFF_SECONDS,
) -> Path:
  """Download one file, retrying transient failures; each retry resumes from the bytes already on disk."""
  with _exclusive(destination):
    if destination.exists():
      verify_pinned_file(pinned, destination)
      return destination

    partial = destination.with_name(destination.name + '.part')
    for attempt in range(1, attempts + 1):
      try:
        _download(f'{base_url}/{pinned.path}', partial, pinned.size, client)
        break
      except (httpx.TransportError, _RetryableStatusError):
        if attempt == attempts:
          raise
        time.sleep(backoff_seconds * 2 ** (attempt - 1))

    try:
      verify_pinned_file(pinned, partial)
    except PinnedFileMismatchError:
      partial.unlink(missing_ok=True)
      raise
    partial.replace(destination)
    return destination


def verify_pinned_files(source: BenchmarkSource, raw_dir: Path) -> dict[str, Path]:
  """The local copies of ``source``'s files, after checking them against their pins."""
  paths = {role: raw_dir / pinned.path for role, pinned in source.files.items()}
  for role, pinned in source.files.items():
    verify_pinned_file(pinned, paths[role])
  return paths


def verify_pinned_file(pinned: PinnedFile, path: Path) -> None:
  if not path.exists():
    raise PinnedFileMismatchError(f'{path} is missing; run `syllo-exp benchmarks fetch` first')
  size = path.stat().st_size
  if size != pinned.size:
    raise PinnedFileMismatchError(f'{path} has {size} bytes, expected {pinned.size}')
  digest = file_sha256(path)
  if digest != pinned.sha256:
    raise PinnedFileMismatchError(f'{path} has SHA-256 {digest}, expected {pinned.sha256}')


@contextmanager
def _exclusive(destination: Path) -> Iterator[None]:
  """Hold an exclusive lock on a sibling file that is never renamed or deleted, so every fetch locks the same file."""
  destination.parent.mkdir(parents=True, exist_ok=True)
  with destination.with_name(destination.name + '.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      raise FetchInProgressError(f'Another fetch is already downloading {destination}') from None
    yield


def _download(url: str, partial: Path, size: int, client: httpx.Client) -> None:
  offset = partial.stat().st_size if partial.exists() else 0
  if offset > size:
    partial.unlink()
    offset = 0
  if offset == size:
    return
  headers = {**_HEADERS, 'Range': f'bytes={offset}-'} if offset else _HEADERS
  with client.stream('GET', url, headers=headers, follow_redirects=True) as response:
    if response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE and offset:
      # The partial file does not match what the server holds: start over.
      partial.unlink()
      raise _RetryableStatusError('Range not satisfiable', request=response.request, response=response)
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS or response.is_server_error:
      raise _RetryableStatusError(f'{url} answered {response.status_code}', request=response.request, response=response)
    response.raise_for_status()
    # A server that ignores the range sends the whole file again.
    resumed = offset > 0 and response.status_code == httpx.codes.PARTIAL_CONTENT
    with partial.open('ab' if resumed else 'wb') as file:
      for chunk in response.iter_bytes(_CHUNK_BYTES):
        file.write(chunk)
