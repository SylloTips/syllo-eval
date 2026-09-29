import logging
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from syllo_eval.datasets.ground_truths import build_sample_ground_truths
from syllo_eval.datasets.model import DatasetJsonPayload
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import TransactionalUnitOfWork, UnitOfWork
from syllo_eval.model import Dataset, DatasetSummary, GroundTruth, Sample

logger = logging.getLogger(__name__)


class DatasetAlreadyExistsError(ValueError):
  """Raised when importing a dataset whose name is already registered."""


@dataclass(slots=True)
class DatasetImportSummary:
  dataset: Dataset
  sample_count: int
  ground_truth_count: int


@dataclass(slots=True)
class DatasetListPage:
  items: Sequence[DatasetSummary]
  total: int
  limit: int
  offset: int


class DatasetService:
  def __init__(self, db_manager: DatabaseManager):
    self._db_manager = db_manager

  async def initialize(self) -> None:
    await self._db_manager.initialize_async()
    if not await self._db_manager.health_check():
      raise RuntimeError('Database health check failed')

  async def close(self) -> None:
    await self._db_manager.close_async()

  async def import_dataset(self, *, name: str, payload: DatasetJsonPayload) -> DatasetImportSummary:
    """Import a dataset payload as a new dataset with its samples and derived ground truths."""
    async with TransactionalUnitOfWork(self._db_manager) as uow:
      existing_dataset = await uow.datasets.get_by_name(name)
      if existing_dataset is not None:
        raise DatasetAlreadyExistsError(f"Dataset '{name}' already exists.")

      dataset = await uow.datasets.create(Dataset(id=uuid4(), name=name))
      created_samples = await uow.samples.bulk_create(
        [
          Sample(
            id=uuid4(),
            dataset_id=dataset.id,
            input_prompt=item.input_prompt,
            ground_truth_output=item.ground_truth_output,
          )
          for item in payload.samples
        ]
      )

      ground_truths: list[GroundTruth] = []
      for created_sample, dataset_sample in zip(created_samples, payload.samples, strict=True):
        ground_truths.extend(build_sample_ground_truths(created_sample.id, dataset_sample))

      created_ground_truths = await uow.ground_truths.bulk_create(ground_truths)

    logger.info(
      'Imported dataset name=%s samples=%s ground_truths=%s', name, len(created_samples), len(created_ground_truths)
    )
    return DatasetImportSummary(
      dataset=dataset,
      sample_count=len(created_samples),
      ground_truth_count=len(created_ground_truths),
    )

  async def list_datasets(self, *, limit: int, offset: int) -> DatasetListPage:
    """List registered datasets in reverse chronological order with their sample counts."""
    async with UnitOfWork(self._db_manager) as uow:
      items = await uow.datasets.list_with_sample_counts(limit=limit, offset=offset)
      total = await uow.datasets.count_all()

    return DatasetListPage(items=items, total=total, limit=limit, offset=offset)
