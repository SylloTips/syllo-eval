from uuid import UUID, uuid4

from syllo_eval.datasets.model import DatasetJsonSample
from syllo_eval.model import GroundTruth, GroundTruthKey


def build_sample_ground_truths(sample_id: UUID, dataset_sample: DatasetJsonSample) -> list[GroundTruth]:
  """Build one ground-truth row per ground-truth kind available on a dataset sample."""
  ground_truths = [
    GroundTruth(
      id=uuid4(),
      sample_id=sample_id,
      key=GroundTruthKey.RELEVANT_DOCUMENT_IDS.value,
      ground_truth_value={'relevant_ids': list(dataset_sample.document_ids)},
    ),
    GroundTruth(
      id=uuid4(),
      sample_id=sample_id,
      key=GroundTruthKey.RELEVANT_SNIPPET_IDS.value,
      ground_truth_value={'relevant_ids': list(dataset_sample.snippet_ids)},
    ),
  ]

  expected_output = (dataset_sample.ground_truth_output or '').strip()
  if expected_output:
    ground_truths.append(
      GroundTruth(
        id=uuid4(),
        sample_id=sample_id,
        key=GroundTruthKey.EXPECTED_OUTPUT.value,
        ground_truth_value={'expected_output': expected_output},
      )
    )

  if dataset_sample.plan:
    ground_truths.append(
      GroundTruth(
        id=uuid4(),
        sample_id=sample_id,
        key=GroundTruthKey.EXPECTED_PLAN.value,
        ground_truth_value={'expected_plan': [step.model_dump(exclude_none=True) for step in dataset_sample.plan]},
      )
    )

  if dataset_sample.claims:
    ground_truths.append(
      GroundTruth(
        id=uuid4(),
        sample_id=sample_id,
        key=GroundTruthKey.EXPECTED_CLAIMS.value,
        ground_truth_value={'claims': [claim.model_dump() for claim in dataset_sample.claims]},
      )
    )

  return ground_truths
