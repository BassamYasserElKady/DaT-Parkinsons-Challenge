#!/usr/bin/env python3

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "submission_src_ensemble" / "assets" / "template.nii"
ENSEMBLE_DIR = ROOT / "results" / "ants_registration_mni"
FLIRT_DIR = ROOT / "results" / "fsl_flirt_affine12" / "registered"
OUT_DIR = ROOT / "results" / "fsl_flirt_affine12" / "dice_comparison"


def dice(mask_a, mask_b):
    denominator = int(mask_a.sum()) + int(mask_b.sum())
    return 2.0 * int(np.logical_and(mask_a, mask_b).sum()) / denominator if denominator else float("nan")


def evaluate(uid):
    template = np.asarray(nib.load(str(TEMPLATE_PATH)).dataobj) > 0
    ensemble = np.asarray(nib.load(str(ENSEMBLE_DIR / f"{uid}.nii.gz")).dataobj) > 0
    flirt = np.asarray(nib.load(str(FLIRT_DIR / f"{uid}.nii.gz")).dataobj) > 0
    ensemble_dice = dice(ensemble, template)
    flirt_dice = dice(flirt, template)
    return {
        "uid": uid,
        "ensemble_ants_rigid_dice": ensemble_dice,
        "fsl_flirt_affine12_dice": flirt_dice,
        "flirt_minus_ensemble": flirt_dice - ensemble_dice,
        "ensemble_foreground_voxels": int(ensemble.sum()),
        "flirt_foreground_voxels": int(flirt.sum()),
        "template_foreground_voxels": int(template.sum()),
    }


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "maximum": float(values.max()),
    }


def main():
    ensemble_uids = {path.name.removesuffix(".nii.gz") for path in ENSEMBLE_DIR.glob("*.nii.gz")}
    flirt_uids = {path.name.removesuffix(".nii.gz") for path in FLIRT_DIR.glob("*.nii.gz")}
    uids = sorted(ensemble_uids & flirt_uids)
    if len(uids) != 1362:
        raise RuntimeError(f"Expected 1,362 paired scans, found {len(uids)}")

    with ProcessPoolExecutor(max_workers=min(16, len(uids))) as executor:
        rows = list(executor.map(evaluate, uids, chunksize=8))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "per_scan_dice.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ensemble = np.asarray([row["ensemble_ants_rigid_dice"] for row in rows])
    flirt = np.asarray([row["fsl_flirt_affine12_dice"] for row in rows])
    difference = flirt - ensemble
    summary = {
        "metric": "positive-voxel foreground Dice against the DaT-SPECT template",
        "paired_scans": len(rows),
        "ensemble_ants_rigid": describe(ensemble),
        "fsl_flirt_affine12": describe(flirt),
        "paired_difference_flirt_minus_ensemble": describe(difference),
        "flirt_higher_count": int((difference > 0).sum()),
        "equal_count": int((difference == 0).sum()),
        "ensemble_higher_count": int((difference < 0).sum()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
