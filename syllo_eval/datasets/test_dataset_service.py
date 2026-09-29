import unittest
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

from syllo_eval.datasets.model import DatasetJsonPayload, DatasetJsonPlanStep, DatasetJsonSample
from syllo_eval.datasets.service import DatasetAlreadyExistsError, DatasetService
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.model import Dataset, DatasetSummary


class _DatasetRepoStub:
  def __init__(self, existing: Dataset | None = None):
    self.existing = existing
    self.created: list[Dataset] = []
    self.list_calls: list[dict[str, int]] = []
    self.summaries: list[DatasetSummary] = []
    self.total = 0

  async def get_by_name(self, name: str) -> Dataset | None:
    del name
    return self.existing

  async def create(self, dataset: Dataset) -> Dataset:
    self.created.append(dataset)
    return dataset

  async def list_with_sample_counts(self, limit: int, offset: int = 0) -> list[DatasetSummary]:
    self.list_calls.append({'limit': limit, 'offset': offset})
    return self.summaries

  async def count_all(self) -> int:
    return self.total


class _BulkCreateRepoStub:
  def __init__(self):
    self.created: list[Any] = []

  async def bulk_create(self, entities: list[Any]) -> list[Any]:
    self.created = list(entities)
    return self.created


class _UnitOfWorkStub:
  def __init__(self, datasets: _DatasetRepoStub):
    self.datasets = datasets
    self.samples = _BulkCreateRepoStub()
    self.ground_truths = _BulkCreateRepoStub()

  async def __aenter__(self) -> '_UnitOfWorkStub':
    return self

  async def __aexit__(self, exc_type, exc, tb) -> bool:
    return False


def _build_service() -> DatasetService:
  return DatasetService(db_manager=cast(DatabaseManager, object()))


def _full_sample() -> DatasetJsonSample:
  return DatasetJsonSample(
    input_prompt='What is the capital of France?',
    ground_truth_output='The share capital is 10000 euros.',
    snippet_ids=['snippet-1'],
    document_ids=['document-1'],
    plan=[DatasetJsonPlanStep(operation='answer', instruction='answer the question')],
  )


class DatasetServiceImportTest(unittest.IsolatedAsyncioTestCase):
  async def test_import_dataset_creates_dataset_samples_and_ground_truths(self) -> None:
    service = _build_service()
    uow = _UnitOfWorkStub(_DatasetRepoStub())
    payload = DatasetJsonPayload(samples=[_full_sample()])

    with patch('syllo_eval.datasets.service.TransactionalUnitOfWork', return_value=uow):
      summary = await service.import_dataset(name='demo', payload=payload)

    self.assertEqual(len(uow.datasets.created), 1)
    self.assertEqual(uow.datasets.created[0].name, 'demo')
    self.assertEqual(summary.dataset.name, 'demo')
    self.assertEqual(summary.sample_count, 1)
    self.assertEqual(len(uow.samples.created), 1)
    self.assertEqual(uow.samples.created[0].dataset_id, summary.dataset.id)
    self.assertEqual(summary.ground_truth_count, 4)
    self.assertEqual(len(uow.ground_truths.created), 4)

  async def test_import_dataset_skips_optional_ground_truths(self) -> None:
    service = _build_service()
    uow = _UnitOfWorkStub(_DatasetRepoStub())
    payload = DatasetJsonPayload(samples=[DatasetJsonSample(input_prompt='Question without expectations')])

    with patch('syllo_eval.datasets.service.TransactionalUnitOfWork', return_value=uow):
      summary = await service.import_dataset(name='demo', payload=payload)

    self.assertEqual(summary.ground_truth_count, 2)

  async def test_import_dataset_rejects_duplicate_name(self) -> None:
    service = _build_service()
    existing = Dataset(id=uuid4(), name='demo')
    uow = _UnitOfWorkStub(_DatasetRepoStub(existing=existing))

    with patch('syllo_eval.datasets.service.TransactionalUnitOfWork', return_value=uow):
      with self.assertRaises(DatasetAlreadyExistsError):
        await service.import_dataset(name='demo', payload=DatasetJsonPayload(samples=[_full_sample()]))

    self.assertEqual(uow.datasets.created, [])
    self.assertEqual(uow.samples.created, [])
    self.assertEqual(uow.ground_truths.created, [])


class DatasetServiceListTest(unittest.IsolatedAsyncioTestCase):
  async def test_list_datasets_returns_page(self) -> None:
    service = _build_service()
    datasets_repo = _DatasetRepoStub()
    datasets_repo.summaries = [
      DatasetSummary(id=uuid4(), name='demo', created_at=datetime.now(tz=timezone.utc), sample_count=3)
    ]
    datasets_repo.total = 7
    uow = _UnitOfWorkStub(datasets_repo)

    with patch('syllo_eval.datasets.service.UnitOfWork', return_value=uow):
      page = await service.list_datasets(limit=10, offset=5)

    self.assertEqual(datasets_repo.list_calls, [{'limit': 10, 'offset': 5}])
    self.assertEqual(len(page.items), 1)
    self.assertEqual(page.items[0].name, 'demo')
    self.assertEqual(page.total, 7)
    self.assertEqual(page.limit, 10)
    self.assertEqual(page.offset, 5)


if __name__ == '__main__':
  unittest.main()
