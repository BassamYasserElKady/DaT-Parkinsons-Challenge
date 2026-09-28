#!/usr/bin/env python3
"""Constrain each adaptive-eight per-slice contour inside a six-slice MIP guide."""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / "results" / "end_to_end_raw_registration_cv5_seed20260931" / "registered"
BRAIN = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
OUT = ROOT / "results" / "adaptive8_mip6_guided_contours"
SRC = ROOT / "submission_ensemble_sigma075_8slice_svm" / "src"


def build_one(uid: str) -> dict[str, object]:
    sys.path.insert(0, str(SRC))
    from adaptive_localizer import adaptive_localization
    from svm_features import retain_top_bins, segment
    image = nib.load(str(REGISTERED / f"{uid}.nii.gz"))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    best, _ = adaptive_localization(volume)
    center = (int(best["start"]) + int(best["end"])) // 2
    start8 = max(0, min(center - 4, volume.shape[2] - 8))
    start6 = max(0, min(center - 3, volume.shape[2] - 6))
    end6 = start6 + 6
    mip = volume[:, :, start6:end6].max(axis=2)
    brain2 = brain[:, :, start6:end6].any(axis=2)
    guide = segment(retain_top_bins(mip, brain2), brain2)
    contour = np.zeros(volume.shape, dtype=np.uint8); unconstrained = []; constrained = []
    for z in range(start8, start8 + 8):
        per_slice = segment(retain_top_bins(volume[:, :, z], brain[:, :, z]), brain[:, :, z])
        guided = per_slice & guide
        contour[:, :, z] = guided
        unconstrained.append(int(per_slice.sum())); constrained.append(int(guided.sum()))
    output = OUT / "contours" / f"{uid}_adaptive8_contours.nii.gz"
    header = image.header.copy(); header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(contour, image.affine, header), str(output))
    return {"uid": uid, "adaptive8_start": start8, "adaptive8_end": start8 + 7,
            "mip6_start": start6, "mip6_end": end6 - 1, "guide_voxels": int(guide.sum()),
            "unconstrained_voxels": int(sum(unconstrained)), "guided_voxels": int(sum(constrained)),
            "guided_nonempty_slices": int(sum(x > 0 for x in constrained)),
            **{f"slice_{i}_guided_voxels": n for i, n in enumerate(constrained)}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "contours").mkdir(exist_ok=True)
    uids = [row["uid"] for row in csv.DictReader(LABELS.open())]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(build_one, uid) for uid in uids]
        for n, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if n % 50 == 0 or n == len(uids): print(f"guided_contours={n}/{len(uids)}", flush=True)
    pd.DataFrame(rows).sort_values("uid").to_csv(OUT / "contour_audit.csv", index=False)


if __name__ == "__main__": main()
