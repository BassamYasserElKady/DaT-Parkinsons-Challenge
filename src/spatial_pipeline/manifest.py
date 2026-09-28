"""Manifest parsing and sample-level split validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "split",
    "deepspot_path",
    "transcripts_path",
    "he_full_width_px",
    "he_full_height_px",
    "he_small_width_px",
    "he_small_height_px",
    "tile_size_px",
    "xenium_um_per_pixel",
    "affine_he_small_to_xenium",
)
ALLOWED_SPLITS = {"development", "final_test"}


@dataclass(frozen=True)
class SampleRecord:
    """Validated, sample-specific inputs and registration metadata."""

    sample_id: str
    split: str
    deepspot_path: Path
    transcripts_path: Path
    he_full_width_px: int
    he_full_height_px: int
    he_small_width_px: int
    he_small_height_px: int
    tile_size_px: int
    xenium_um_per_pixel: float
    affine_he_small_to_xenium: np.ndarray

    @property
    def is_final_test(self) -> bool:
        return self.split == "final_test"


def _parse_affine(value: str) -> np.ndarray:
    parts = [part.strip() for part in str(value).split(";")]
    if len(parts) != 9:
        raise ValueError(
            "affine_he_small_to_xenium must contain 9 semicolon-separated values"
        )
    try:
        matrix = np.asarray([float(part) for part in parts], dtype=np.float64).reshape(3, 3)
    except ValueError as exc:
        raise ValueError("affine transform contains a non-numeric value") from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError("affine transform must contain finite values")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("affine transform must be invertible")
    return matrix


def _positive_int(row: pd.Series, name: str) -> int:
    try:
        value = int(row[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(row: pd.Series, name: str) -> float:
    try:
        value = float(row[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _resolve_relative_path(root: Path, value: str, field: str) -> Path:
    path = Path(str(value).strip())
    if not str(value).strip() or path.is_absolute():
        raise ValueError(f"{field} must be a non-empty relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the project root") from exc
    return resolved


def validate_manifest(frame: pd.DataFrame, root: Path, expected_development: int | None = None, expected_final_test: int | None = None) -> list[SampleRecord]:
    """Validate a manifest frame and return immutable sample records.

    Validation is schema- and metadata-only; input data files are not opened here.
    """

    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    if frame.empty:
        raise ValueError("manifest must contain at least one sample")

    records: list[SampleRecord] = []
    seen: set[str] = set()
    for index, row in frame.iterrows():
        sample_id = str(row["sample_id"]).strip()
        split = str(row["split"]).strip()
        if not sample_id or sample_id.lower() == "nan":
            raise ValueError(f"row {index}: sample_id is required")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"row {index}: split must be one of {sorted(ALLOWED_SPLITS)}")

        records.append(
            SampleRecord(
                sample_id=sample_id,
                split=split,
                deepspot_path=_resolve_relative_path(root, row["deepspot_path"], "deepspot_path"),
                transcripts_path=_resolve_relative_path(root, row["transcripts_path"], "transcripts_path"),
                he_full_width_px=_positive_int(row, "he_full_width_px"),
                he_full_height_px=_positive_int(row, "he_full_height_px"),
                he_small_width_px=_positive_int(row, "he_small_width_px"),
                he_small_height_px=_positive_int(row, "he_small_height_px"),
                tile_size_px=_positive_int(row, "tile_size_px"),
                xenium_um_per_pixel=_positive_float(row, "xenium_um_per_pixel"),
                affine_he_small_to_xenium=_parse_affine(row["affine_he_small_to_xenium"]),
            )
        )

    development = sum(record.split == "development" for record in records)
    final_test = sum(record.split == "final_test" for record in records)
    if expected_development is not None and development != expected_development:
        raise ValueError(f"manifest must contain exactly {expected_development} development samples; found {development}")
    if expected_final_test is not None and final_test != expected_final_test:
        raise ValueError(f"manifest must contain exactly {expected_final_test} final_test samples; found {final_test}")
    return records


def load_manifest(path: Path, root: Path, expected_development: int | None = None, expected_final_test: int | None = None) -> list[SampleRecord]:
    """Load and validate a CSV manifest without opening any input data files."""

    frame = pd.read_csv(path)
    return validate_manifest(frame, root, expected_development=expected_development, expected_final_test=expected_final_test)


def assert_sample_disjoint(*groups: Iterable[str]) -> None:
    """Raise if any sample ID occurs in more than one split/group."""

    owners: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        for sample_id in group:
            if sample_id in owners:
                raise ValueError(
                    f"sample-level leakage: {sample_id!r} occurs in groups "
                    f"{owners[sample_id]} and {group_index}"
                )
            owners[sample_id] = group_index


def grouped_validation_folds(
    records: Sequence[SampleRecord], n_splits: int = 5, seed: int = 42
) -> list[tuple[list[str], list[str]]]:
    """Create deterministic sample-grouped folds within development samples."""

    development_ids = sorted(record.sample_id for record in records if record.split == "development")
    if n_splits < 2 or n_splits > len(development_ids):
        raise ValueError("n_splits must be between 2 and the number of development samples")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(development_ids, dtype=object)
    rng.shuffle(shuffled)
    folds = [list(shuffled[index::n_splits]) for index in range(n_splits)]
    result: list[tuple[list[str], list[str]]] = []
    for validation in folds:
        validation_set = set(validation)
        train = [sample_id for sample_id in development_ids if sample_id not in validation_set]
        assert_sample_disjoint(train, validation)
        result.append((train, validation))
    return result

