#!/usr/bin/env python3
"""Derive bilateral 3D contour shapes and four centroid-based quarters per side."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTOURS = ROOT / "results" / "adaptive8_per_slice_contours" / "contours"
OUT = ROOT / "results" / "adaptive8_3d_quarter_shapes"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
MIDLINE_X = 45


def quarter_labels(mask: np.ndarray, base_label: int) -> tuple[np.ndarray, list[int]]:
    """Split one irregular contour using two PCA planes through its centroid."""
    result = np.zeros(mask.shape, dtype=np.uint8)
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return result, [0, 0, 0, 0]
    center = coordinates.mean(axis=0)
    if len(coordinates) >= 3:
        _, _, vectors = np.linalg.svd(coordinates - center, full_matrices=False)
        # The sign of a PCA vector is arbitrary.  Fix it for reproducible labels.
        axis_a, axis_b = vectors[0], vectors[1]
        if axis_a[np.argmax(np.abs(axis_a))] < 0:
            axis_a = -axis_a
        if axis_b[np.argmax(np.abs(axis_b))] < 0:
            axis_b = -axis_b
        a = (coordinates - center) @ axis_a
        b = (coordinates - center) @ axis_b
    else:
        # Degenerate tiny contour: still split reproducibly around its centroid.
        a, b = coordinates[:, 1] - center[1], coordinates[:, 2] - center[2]
    quadrant = (a >= 0).astype(np.uint8) + 2 * (b >= 0).astype(np.uint8)
    for q in range(4):
        chosen = coordinates[quadrant == q]
        if len(chosen):
            result[tuple(chosen.T)] = base_label + q
    return result, [int(np.sum(quadrant == q)) for q in range(4)]


def build_one(uid: str) -> dict[str, object]:
    image = nib.load(str(CONTOURS / f"{uid}_adaptive8_contours.nii.gz"))
    contour = np.asarray(image.dataobj) > 0
    x_grid = np.indices(contour.shape, sparse=True)[0]
    left, right = contour & (x_grid >= MIDLINE_X), contour & (x_grid < MIDLINE_X)
    labels_left, left_counts = quarter_labels(left, base_label=1)
    labels_right, right_counts = quarter_labels(right, base_label=5)
    labels = labels_left + labels_right
    output = OUT / "labels" / f"{uid}_adaptive8_3d_quarters.nii.gz"
    header = image.header.copy(); header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(labels, image.affine, header), str(output))
    return {
        "uid": uid,
        "left_shape_voxels": int(left.sum()), "right_shape_voxels": int(right.sum()),
        **{f"left_quarter_{q + 1}_voxels": value for q, value in enumerate(left_counts)},
        **{f"right_quarter_{q + 1}_voxels": value for q, value in enumerate(right_counts)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(exist_ok=True)
    uids = [row["uid"] for row in csv.DictReader(LABELS.open())]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(build_one, uid): uid for uid in uids}
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 50 == 0 or count == len(uids):
                print(f"shapes={count}/{len(uids)}", flush=True)
    audit = pd.DataFrame(rows).sort_values("uid")
    audit.to_csv(OUT / "shape_audit.csv", index=False)
    labels = {
        "0": "outside contour", "1": "left quarter 1", "2": "left quarter 2",
        "3": "left quarter 3", "4": "left quarter 4", "5": "right quarter 1",
        "6": "right quarter 2", "7": "right quarter 3", "8": "right quarter 4",
    }
    (OUT / "README.json").write_text(json.dumps({
        "full_left_shape": "union of labels 1-4", "full_right_shape": "union of labels 5-8",
        "split": "two PCA planes crossing at each side's 3D contour centroid; no equal-volume constraint",
        "labels": labels, "scan_count": len(uids),
    }, indent=2) + "\n")
    print(audit.drop(columns="uid").describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
