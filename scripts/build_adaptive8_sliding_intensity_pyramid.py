#!/usr/bin/env python3
"""Build sparse sliding-cube intensity maps on the eight-slice contours.

For every in-contour voxel (the anchor), this records its raw registered SPECT
intensity and same-side contour-only local means in cubes whose lower corner is
the anchor: 1x1x1, 2x2x2, and 3x3x3.  The sparse arrays preserve MNI voxel
coordinates without allocating six mostly empty full volumes per scan.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import convolve

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / "results" / "end_to_end_raw_registration_cv5_seed20260931" / "registered"
CONTOURS = ROOT / "results" / "adaptive8_per_slice_contours" / "contours"
OUT = ROOT / "results" / "adaptive8_sliding_intensity_pyramid"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
MIDLINE_X = 45  # Same x-axis bilateral split used by the existing SVM extractor.


def sliding_mean(image: np.ndarray, side_mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Contour-only sliding cube mean and the number of contributing voxels.

    ``origin=0`` and the all-ones kernel define a reproducible anchored grid;
    at each coordinate, the cube includes that coordinate and its positive-axis
    neighbours.  No intensity outside the same side's contour is included.
    """
    kernel = np.ones((size, size, size), dtype=np.float32)
    numerator = convolve(np.where(side_mask, image, 0.0), kernel, mode="constant", cval=0.0, origin=0)
    support = convolve(side_mask.astype(np.float32), kernel, mode="constant", cval=0.0, origin=0)
    return np.divide(numerator, support, out=np.zeros_like(numerator, dtype=np.float32), where=support > 0), support.astype(np.uint8)


def build_one(uid: str) -> dict[str, object]:
    image = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
    contour = np.asarray(nib.load(str(CONTOURS / f"{uid}_adaptive8_contours.nii.gz")).dataobj) > 0
    if image.shape != contour.shape:
        raise RuntimeError(f"GRID_MISMATCH:{uid}:{image.shape}!={contour.shape}")
    grid_x = np.indices(contour.shape, sparse=True)[0]
    left = contour & (grid_x >= MIDLINE_X)
    right = contour & (grid_x < MIDLINE_X)
    # Keep all anchors in a single coordinate table; side=0 left, side=1 right.
    coords_left = np.argwhere(left)
    coords_right = np.argwhere(right)
    coords = np.vstack((coords_left, coords_right)).astype(np.int16, copy=False)
    side = np.concatenate((np.zeros(len(coords_left), dtype=np.uint8), np.ones(len(coords_right), dtype=np.uint8)))
    raw = image[tuple(coords.T)].astype(np.float32, copy=False) if len(coords) else np.empty(0, np.float32)
    level2, support2 = sliding_mean(image, left, 2)
    left_l2 = level2[tuple(coords_left.T)] if len(coords_left) else np.empty(0, np.float32)
    left_s2 = support2[tuple(coords_left.T)] if len(coords_left) else np.empty(0, np.uint8)
    level2, support2 = sliding_mean(image, right, 2)
    right_l2 = level2[tuple(coords_right.T)] if len(coords_right) else np.empty(0, np.float32)
    right_s2 = support2[tuple(coords_right.T)] if len(coords_right) else np.empty(0, np.uint8)
    level3, support3 = sliding_mean(image, left, 3)
    left_l3 = level3[tuple(coords_left.T)] if len(coords_left) else np.empty(0, np.float32)
    left_s3 = support3[tuple(coords_left.T)] if len(coords_left) else np.empty(0, np.uint8)
    level3, support3 = sliding_mean(image, right, 3)
    right_l3 = level3[tuple(coords_right.T)] if len(coords_right) else np.empty(0, np.float32)
    right_s3 = support3[tuple(coords_right.T)] if len(coords_right) else np.empty(0, np.uint8)
    np.savez_compressed(
        OUT / "maps" / f"{uid}_adaptive8_sliding_pyramid.npz",
        xyz=coords, side=side, intensity_1=raw,
        intensity_2=np.concatenate((left_l2, right_l2)).astype(np.float32),
        intensity_3=np.concatenate((left_l3, right_l3)).astype(np.float32),
        support_2=np.concatenate((left_s2, right_s2)),
        support_3=np.concatenate((left_s3, right_s3)),
    )
    record: dict[str, object] = {
        "uid": uid, "anchor_voxels_total": int(len(coords)),
        "left_anchor_voxels": int(len(coords_left)), "right_anchor_voxels": int(len(coords_right)),
    }
    for name, values in (("level1", raw), ("level2", np.concatenate((left_l2, right_l2))), ("level3", np.concatenate((left_l3, right_l3)))):
        record[f"{name}_mean"] = float(values.mean()) if len(values) else 0.0
        record[f"{name}_std"] = float(values.std()) if len(values) else 0.0
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "maps").mkdir(exist_ok=True)
    uids = [row["uid"] for row in csv.DictReader(LABELS.open())]
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(build_one, uid): uid for uid in uids}
        for count, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if count % 50 == 0 or count == len(uids):
                print(f"maps={count}/{len(uids)}", flush=True)
    audit = pd.DataFrame(records).sort_values("uid")
    audit.to_csv(OUT / "map_audit.csv", index=False)
    metadata = {
        "maps": "sparse per-scan NPZ files",
        "coordinate_order": "xyz in registered MNI grid",
        "side": "0=left, 1=right; follows the existing SVM x=45 split",
        "intensity_1": "raw registered intensity at anchor voxel (1x1x1)",
        "intensity_2": "same-side contour-only local mean in anchored overlapping 2x2x2 cube",
        "intensity_3": "same-side contour-only local mean in anchored overlapping 3x3x3 cube",
        "support_2/support_3": "number of contour voxels contributing to each local mean",
        "scan_count": len(uids),
    }
    (OUT / "README.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(audit[["anchor_voxels_total", "left_anchor_voxels", "right_anchor_voxels"]].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
