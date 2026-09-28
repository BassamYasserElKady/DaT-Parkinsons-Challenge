#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

import ants
import numpy as np

from submission_src_ensemble.src.adaptive_localizer import adaptive_localization


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "external" / "niftis_original"
TEMPLATE = ROOT / "submission_src_ensemble" / "assets" / "template.nii"
OUTPUT = ROOT / "results" / "rigid_candidate_quality"
CONFIGS = {
    "rigid_default": {"type_of_transform": "Rigid"},
    "dense_rigid": {"type_of_transform": "DenseRigid"},
    "rigid_high_sampling": {"type_of_transform": "Rigid", "aff_sampling": 64, "aff_random_sampling_rate": 0.5},
}


def score_one(path_string: str):
    fixed = ants.image_read(str(TEMPLATE))
    fixed_array = np.asarray(fixed.numpy())
    moving = ants.image_read(path_string)
    uid = Path(path_string).name.removesuffix(".nii.gz")
    rows = []
    for candidate, kwargs in CONFIGS.items():
        result = ants.registration(fixed=fixed, moving=moving, verbose=False, **kwargs)
        warped = result["warpedmovout"]
        array = np.asarray(warped.numpy())
        window, _ = adaptive_localization(array)
        rows.append({
            "uid": uid,
            "candidate": candidate,
            "mutual_information": float(ants.image_mutual_information(fixed, warped)),
            "foreground_overlap": float(np.mean((array > 0) & (fixed_array > 0))),
            "localized_slices": int(window["detected"]),
            "localizer_score": float(window["score"]),
        })
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT.glob("*.nii.gz"))
    if len(paths) != 1362:
        raise RuntimeError(f"Expected 1362 scans, found {len(paths)}")
    rows = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for index, result in enumerate(executor.map(score_one, map(str, paths)), start=1):
            rows.extend(result)
            if index % 25 == 0 or index == len(paths):
                print(f"Scored {index}/{len(paths)} scans", flush=True)
    with (OUTPUT / "candidate_quality.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
