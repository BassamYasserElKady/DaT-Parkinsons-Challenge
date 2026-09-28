#!/usr/bin/env python3
"""Recursively divide each of the bilateral four-quarter shapes into four parts."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results" / "adaptive8_mip6_guided_3d_quarter_shapes" / "labels"
OUT = ROOT / "results" / "adaptive8_mip6_guided_hierarchical_quarters"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"


def split_four(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    output = np.zeros(mask.shape, dtype=np.uint8); xyz = np.argwhere(mask)
    if not len(xyz): return output, [0, 0, 0, 0]
    center = xyz.mean(axis=0)
    if len(xyz) >= 3:
        _, _, vectors = np.linalg.svd(xyz - center, full_matrices=False)
        a_axis, b_axis = vectors[:2]
        if a_axis[np.argmax(np.abs(a_axis))] < 0: a_axis = -a_axis
        if b_axis[np.argmax(np.abs(b_axis))] < 0: b_axis = -b_axis
        a, b = (xyz - center) @ a_axis, (xyz - center) @ b_axis
    else:
        a, b = xyz[:, 1] - center[1], xyz[:, 2] - center[2]
    group = (a >= 0).astype(np.uint8) + 2 * (b >= 0).astype(np.uint8)
    for part in range(4):
        chosen = xyz[group == part]
        if len(chosen): output[tuple(chosen.T)] = part + 1
    return output, [int(np.sum(group == part)) for part in range(4)]


def one(uid: str) -> dict[str, object]:
    image = nib.load(str(PARENT / f"{uid}_adaptive8_3d_quarters.nii.gz"))
    parent = np.asarray(image.dataobj, dtype=np.uint8)
    # 1-4 first-level left; 5-8 first-level right.  New labels 9-24 left,
    # 25-40 right, ordered parent quarter then subquarter.
    child = np.zeros(parent.shape, dtype=np.uint8); record: dict[str, object] = {"uid": uid}
    for side, parent_base, child_base, prefix in (("left", 1, 9, "left"), ("right", 5, 25, "right")):
        for q in range(4):
            local, counts = split_four(parent == parent_base + q)
            for sub in range(4): child[local == sub + 1] = child_base + q * 4 + sub
            for sub, count in enumerate(counts): record[f"{prefix}_quarter_{q+1}_subquarter_{sub+1}_voxels"] = count
    combined = parent.copy()
    combined[child > 0] = child[child > 0]
    header = image.header.copy(); header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(combined, image.affine, header), str(OUT / "labels" / f"{uid}_adaptive8_hierarchical_quarters.nii.gz"))
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(exist_ok=True)
    uids = [r["uid"] for r in csv.DictReader(LABELS.open())]; records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, uid) for uid in uids]
        for n, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if n % 50 == 0 or n == len(uids): print(f"hierarchical_shapes={n}/{len(uids)}", flush=True)
    pd.DataFrame(records).sort_values("uid").to_csv(OUT / "shape_audit.csv", index=False)
    (OUT / "README.json").write_text(json.dumps({
        "full_left": "all labels 1-24", "full_right": "all labels 25-40 plus parent labels 5-8 if present",
        "first_level": "labels 1-4 left, 5-8 right only exist where a child region is absent; reconstruct using child ranges",
        "second_level": "labels 9-24 left and 25-40 right; four subquarters per first-level quarter",
        "note": "For unambiguous model input, reconstruct parent quarters by fixed child-label groups."}, indent=2)+"\n")


if __name__ == "__main__": main()
