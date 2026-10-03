import fcntl
import hashlib
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

from benchmarks.download import (
  FetchInProgressError,
  PinnedFileMismatchError,
  fetch_pinned_file,
  verify_pinned_files,
)
from config import BenchmarkSource, PinnedFile

CONTENT = b'0123456789' * 100
BASE_URL = 'https://example.org/resolve/' + 'a' * 40
PINNED = PinnedFile(path='data/file.bin', size=len(CONTENT), sha256=hashlib.sha256(CONTENT).hexdigest())


class _BrokenStream(httpx.SyncByteStream):
  """Sends some bytes, then fails as a dropped connection would."""

  def __init__(self, sent: bytes):
    self._sent = sent

  def __iter__(self) -> Iterator[bytes]:
    yield self._sent
    raise httpx.ReadError('connection dropped')


class _Server:
  """Serves CONTENT, honouring Range headers unless told to ignore them."""

  def __init__(self, *, content: bytes = CONTENT, honour_ranges: bool = True):
    self.content = content
    self.honour_ranges = honour_ranges
    self.requests: list[httpx.Request] = []

  def __call__(self, request: httpx.Request) -> httpx.Response:
    self.requests.append(request)
    assert request.url == f'{BASE_URL}/{PINNED.path}'
    range_header = request.headers.get('range')
    if range_header and self.honour_ranges:
      start = int(range_header.removeprefix('bytes=').removesuffix('-'))
      return httpx.Response(206, content=self.content[start:])
    return httpx.Response(200, content=self.content)


class FetchPinnedFileTest(unittest.TestCase):
  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.destination = Path(self._directory.name) / PINNED.path
    self.partial = self.destination.with_name(self.destination.name + '.part')

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _fetch(self, server: Callable[[httpx.Request], httpx.Response], attempts: int = 5) -> Path:
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
      return fetch_pinned_file(BASE_URL, PINNED, self.destination, client, attempts=attempts, backoff_seconds=0)

  def test_downloads_and_verifies_a_new_file(self) -> None:
    self.assertEqual(self._fetch(_Server()), self.destination)

    self.assertEqual(self.destination.read_bytes(), CONTENT)
    self.assertFalse(self.partial.exists())

  def test_resumes_a_partial_download_with_a_range_request(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])
    server = _Server()

    self._fetch(server)

    self.assertEqual(server.requests[0].headers['range'], 'bytes=300-')
    self.assertEqual(self.destination.read_bytes(), CONTENT)

  def test_restarts_when_the_server_ignores_the_range(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])

    self._fetch(_Server(honour_ranges=False))

    self.assertEqual(self.destination.read_bytes(), CONTENT)

  def test_rejects_and_discards_bytes_that_do_not_match_the_pin(self) -> None:
    with self.assertRaisesRegex(PinnedFileMismatchError, 'SHA-256'):
      self._fetch(_Server(content=CONTENT[::-1]))

    self.assertFalse(self.destination.exists())
    self.assertFalse(self.partial.exists())

  def test_keeps_a_verified_file_without_downloading_it_again(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.destination.write_bytes(CONTENT)
    server = _Server()

    self._fetch(server)

    self.assertEqual(server.requests, [])

  def test_retries_transient_failures_and_resumes_from_the_partial_file(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])
    server = _Server()
    failures = [httpx.ConnectError('connection reset by peer'), httpx.Response(503), 'broken stream']

    def flaky(request: httpx.Request) -> httpx.Response:
      if not failures:
        return server(request)
      failure = failures.pop(0)
      if isinstance(failure, Exception):
        raise failure
      if isinstance(failure, httpx.Response):
        return failure
      # Bytes still buffered when the connection drops are lost, so the next attempt resumes at the file's end.
      return httpx.Response(206, stream=_BrokenStream(CONTENT[300:400]))

    self._fetch(flaky)

    self.assertEqual(server.requests[0].headers['range'], 'bytes=300-')
    self.assertEqual(self.destination.read_bytes(), CONTENT)

  def test_gives_up_after_the_last_attempt(self) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
      raise httpx.ConnectError('connection refused')

    with self.assertRaises(httpx.ConnectError):
      self._fetch(unreachable, attempts=2)

  def test_client_errors_are_not_retried(self) -> None:
    requests: list[httpx.Request] = []

    def missing(request: httpx.Request) -> httpx.Response:
      requests.append(request)
      return httpx.Response(404)

    with self.assertRaises(httpx.HTTPStatusError):
      self._fetch(missing)
    self.assertEqual(len(requests), 1)

  def test_requests_ask_for_the_stored_bytes_not_a_compressed_encoding(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])
    server = _Server()

    self._fetch(server)

    self.assertEqual(server.requests[0].headers['accept-encoding'], 'identity')

  def test_restarts_from_the_start_when_the_range_cannot_be_satisfied(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])
    server = _Server()
    requests: list[httpx.Request] = []

    def mismatched(request: httpx.Request) -> httpx.Response:
      requests.append(request)
      if 'range' in request.headers:
        return httpx.Response(416)
      return server(request)

    self._fetch(mismatched)

    self.assertEqual([request.headers.get('range') for request in requests], ['bytes=300-', None])
    self.assertEqual(self.destination.read_bytes(), CONTENT)

  def test_a_server_error_that_outlasts_the_retries_is_an_http_status_error(self) -> None:
    with self.assertRaises(httpx.HTTPStatusError):
      self._fetch(lambda request: httpx.Response(503), attempts=2)

  def test_resume_survives_a_cross_host_redirect(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])
    requests: list[httpx.Request] = []

    def redirecting(request: httpx.Request) -> httpx.Response:
      requests.append(request)
      if request.url.host == 'cdn.example.net':
        start = int(request.headers['range'].removeprefix('bytes=').removesuffix('-'))
        return httpx.Response(206, content=CONTENT[start:])
      return httpx.Response(302, headers={'location': 'https://cdn.example.net/signed/file.bin'})

    self._fetch(redirecting)

    self.assertEqual(requests[1].url.host, 'cdn.example.net')
    self.assertEqual(requests[1].headers['range'], 'bytes=300-')
    self.assertEqual(self.destination.read_bytes(), CONTENT)

  def test_a_concurrent_fetch_of_the_same_file_fails_fast_and_leaves_the_partial_alone(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.partial.write_bytes(CONTENT[:300])
    server = _Server()
    with self.destination.with_name(self.destination.name + '.lock').open('a') as lock:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

      with self.assertRaises(FetchInProgressError):
        self._fetch(server)

    self.assertEqual(server.requests, [])
    self.assertEqual(self.partial.read_bytes(), CONTENT[:300])

  def test_a_corrupted_local_copy_is_reported_not_replaced(self) -> None:
    self.destination.parent.mkdir(parents=True)
    self.destination.write_bytes(CONTENT[:10])

    with self.assertRaisesRegex(PinnedFileMismatchError, 'bytes'):
      self._fetch(_Server())


class VerifyPinnedFilesTest(unittest.TestCase):
  def test_missing_files_point_to_the_fetch_step(self) -> None:
    source = BenchmarkSource(dataset_name='kb', expected_samples=1, base_url=BASE_URL, files={'data': PINNED})
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(PinnedFileMismatchError, 'benchmarks fetch'):
        verify_pinned_files(source, Path(directory))


if __name__ == '__main__':
  unittest.main()
