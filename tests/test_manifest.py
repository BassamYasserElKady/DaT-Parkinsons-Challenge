from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_pipeline.manifest import validate_manifest


def valid_frame() -> pd.DataFrame:
    rows = []
    affine = "1;0;0;0;1;0;0;0;1"
    for index in range(14):
        rows.append(
            {
                "sample_id": f"sample_{index + 1:02d}",
                "split": "development" if index < 10 else "final_test",
                "deepspot_path": f"data/raw/sample_{index + 1:02d}/deepspot.parquet",
                "transcripts_path": f"data/raw/sample_{index + 1:02d}/transcripts.parquet",
                "he_full_width_px": 1000,
                "he_full_height_px": 800,
                "he_small_width_px": 500,
                "he_small_height_px": 400,
                "tile_size_px": 224,
                "xenium_um_per_pixel": 0.2125,
                "affine_he_small_to_xenium": affine,
            }
        )
    return pd.DataFrame(rows)


def test_manifest_requires_ten_development_and_four_final_test(tmp_path: Path) -> None:
    records = validate_manifest(valid_frame(), tmp_path)
    assert len(records) == 14
    assert sum(record.split == "development" for record in records) == 10
    assert sum(record.split == "final_test" for record in records) == 4


def test_manifest_rejects_missing_column(tmp_path: Path) -> None:
    frame = valid_frame().drop(columns=["tile_size_px"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_manifest(frame, tmp_path)


def test_manifest_rejects_absolute_input_path(tmp_path: Path) -> None:
    frame = valid_frame()
    frame.loc[0, "deepspot_path"] = "/outside/project.parquet"
    with pytest.raises(ValueError, match="relative path"):
        validate_manifest(frame, tmp_path)
