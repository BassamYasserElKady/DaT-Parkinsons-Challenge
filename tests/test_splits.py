from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_pipeline.manifest import assert_sample_disjoint, grouped_validation_folds
from test_manifest import valid_frame
from spatial_pipeline.manifest import validate_manifest


def test_sample_overlap_raises() -> None:
    with pytest.raises(ValueError, match="sample-level leakage"):
        assert_sample_disjoint(["sample_01", "sample_02"], ["sample_02", "sample_03"])


def test_grouped_validation_folds_are_disjoint(tmp_path: Path) -> None:
    records = validate_manifest(valid_frame(), tmp_path)
    folds = grouped_validation_folds(records, n_splits=5, seed=42)
    assert len(folds) == 5
    for train_ids, validation_ids in folds:
        assert set(train_ids).isdisjoint(validation_ids)
        assert set(train_ids) | set(validation_ids) == {
            f"sample_{index:02d}" for index in range(1, 11)
        }
        assert all(sample_id.startswith("sample_") for sample_id in validation_ids)
