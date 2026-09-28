#!/usr/bin/env python3
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from submission_src_ensemble.src.adaptive_localizer import adaptive_localization


ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / "results" / "rigid_registration_mni"
SPLIT = ROOT / "results" / "holdout_900_120_342_3d_ensemble" / "split.csv"
OUTPUT = ROOT / "results" / "holdout_900_120_342_3d_ensemble" / "train_only_standardized_windows.npy"


def make_raw(path_string: str):
    path = Path(path_string)
    volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    valid = np.isfinite(volume) & (volume > 0)
    values = volume[valid]
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[valid] = (values - values.mean()) / max(values.std(), 1e-8)
    window, _ = adaptive_localization(volume)
    slices = np.stack([normalized[:, :, z].T for z in range(window["start"], window["end"] + 1)])
    return path.name.removesuffix(".nii.gz"), slices


def main():
    rows = list(csv.DictReader(SPLIT.open()))
    ordered_uids = [row["uid"] for row in rows]
    partitions = {row["uid"]: row["partition"] for row in rows}
    paths = [REGISTERED / f"{uid}.nii.gz" for uid in ordered_uids]
    if not all(path.exists() for path in paths):
        raise RuntimeError("Missing registered scans")
    raw = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        for index, (uid, slices) in enumerate(executor.map(make_raw, map(str, paths)), start=1):
            raw[uid] = slices
            if index % 100 == 0 or index == len(paths):
                print(f"Prepared {index}/{len(paths)} windows", flush=True)
    windows = np.stack([raw[uid] for uid in ordered_uids]).astype(np.float32)
    train = np.asarray([partitions[uid] == "train" for uid in ordered_uids])
    means = windows[train].mean(axis=(0, 2, 3), keepdims=True)
    stds = windows[train].std(axis=(0, 2, 3), keepdims=True)
    standardized = (windows - means) / np.maximum(stds, 1e-8)
    target = (16, 64, 64)
    resized = np.stack([ndimage.zoom(item, tuple(target[i] / item.shape[i] for i in range(3)), order=1) for item in standardized]).astype(np.float32)
    np.save(OUTPUT, resized)
    np.savez(OUTPUT.with_suffix('.normalization.npz'), channel_means=means.squeeze(), channel_stds=stds.squeeze())


if __name__ == "__main__":
    main()
