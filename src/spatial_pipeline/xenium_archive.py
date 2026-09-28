"""Read-only access to Xenium arrays stored in a nested ZIP archive.

The downloaded Explorer bundle contains Zarr ZIP files inside an outer ZIP.
This module opens those members in memory and never extracts or modifies the
raw archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import zarr


class NestedZipStore:
    """Context manager for one nested, read-only Zarr ZIP member."""

    def __init__(self, outer: zipfile.ZipFile, member: str):
        self._payload = outer.read(member)
        self._file = BytesIO(self._payload)
        self.store = zarr.storage.ZipStore(self._file, mode="r")

    def __enter__(self):
        return zarr.open_group(self.store, mode="r")

    def __exit__(self, exc_type, exc_value, traceback):
        self.store.close()
        self._file.close()
        self._payload = None


@dataclass
class XeniumCellArrays:
    """Sparse feature-by-cell counts and cell centroid coordinates."""

    feature_names: list[str]
    feature_types: list[str]
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    n_cells: int
    centroids_um: np.ndarray

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def read_cell_arrays(
    archive_path: Path,
    expression_member: str = "cell_feature_matrix.zarr.zip",
    cells_member: str = "cells.zarr.zip",
) -> XeniumCellArrays:
    """Read cell counts and centroids from a Xenium Explorer archive.

    The matrix is the Zarr CSC representation used by Xenium: rows are
    features and columns are cells. Arrays are copied into memory, while the
    raw outer and nested ZIP files remain untouched on disk.
    """

    with zipfile.ZipFile(archive_path, mode="r") as outer:
        with NestedZipStore(outer, expression_member) as expression_root:
            features = expression_root["cell_features"]
            csc = features["csc"]
            feature_names = [_decode(v) for v in features.attrs["feature_keys"]]
            feature_types = [_decode(v) for v in features.attrs["feature_types"]]
            data = np.asarray(csc["data"][:])
            indices = np.asarray(csc["indices"][:])
            indptr = np.asarray(csc["indptr"][:])
            n_cells = int(indptr.shape[0] - 1)

        with NestedZipStore(outer, cells_member) as cells_root:
            summary = cells_root["cell_summary"]
            names = [_decode(v) for v in summary.attrs["column_names"]]
            try:
                x_index = names.index("cell_centroid_x")
                y_index = names.index("cell_centroid_y")
            except ValueError as exc:
                raise ValueError(
                    "cells.zarr.zip lacks cell_centroid_x/cell_centroid_y"
                ) from exc
            centroids_um = np.asarray(summary[:, [x_index, y_index]], dtype=np.float64)

    if centroids_um.shape[0] != n_cells:
        raise ValueError(
            f"cell count mismatch: expression has {n_cells}, "
            f"cells.zarr has {centroids_um.shape[0]}"
        )
    return XeniumCellArrays(
        feature_names=feature_names,
        feature_types=feature_types,
        data=data,
        indices=indices,
        indptr=indptr,
        n_cells=n_cells,
        centroids_um=centroids_um,
    )
