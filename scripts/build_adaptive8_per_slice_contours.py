#!/usr/bin/env python3
"""Create independent bilateral hotspot contours for all adaptive eight-slice slabs.

Each output is a full-grid binary NIfTI mask.  Only the eight selected axial
slices can be nonzero, allowing direct overlay on its registered scan.  No MIP
is used in segmentation.
"""
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
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
BRAIN = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "adaptive8_per_slice_contours"
LOCALIZER_DIR = ROOT / "submission_ensemble_sigma075_8slice_svm" / "src"


def build_one(uid: str) -> dict[str, object]:
    # Imports inside workers keep the process payload small and avoid pickling modules.
    sys.path.insert(0, str(LOCALIZER_DIR))
    from adaptive_localizer import adaptive_localization
    from svm_features import retain_top_bins, segment

    image = nib.load(str(REGISTERED / f"{uid}.nii.gz"))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    if volume.shape != brain.shape:
        raise RuntimeError(f"GRID_MISMATCH:{uid}:{volume.shape}!={brain.shape}")
    localized, _ = adaptive_localization(volume)
    center = (int(localized["start"]) + int(localized["end"])) // 2
    start = max(0, min(center - 4, volume.shape[2] - 8))
    end = start + 8
    contours = np.zeros(volume.shape, dtype=np.uint8)
    counts: list[int] = []
    for z in range(start, end):
        # This is deliberately slice-wise: no max-intensity projection across z.
        mask = segment(retain_top_bins(volume[:, :, z], brain[:, :, z]), brain[:, :, z])
        contours[:, :, z] = mask.astype(np.uint8)
        counts.append(int(mask.sum()))
    output = OUT / "contours" / f"{uid}_adaptive8_contours.nii.gz"
    nib.save(nib.Nifti1Image(contours, image.affine, image.header), str(output))
    return {
        "uid": uid,
        "localizer_start": int(localized["start"]), "localizer_end": int(localized["end"]),
        "adaptive8_start": start, "adaptive8_end": end - 1,
        "nonempty_contour_slices": int(sum(count > 0 for count in counts)),
        "contour_voxels_total": int(sum(counts)),
        **{f"slice_{offset}_z": start + offset for offset in range(8)},
        **{f"slice_{offset}_contour_voxels": count for offset, count in enumerate(counts)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contours").mkdir(exist_ok=True)
    uids = [row["uid"] for row in csv.DictReader(LABELS.open())]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(build_one, uid): uid for uid in uids}
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 50 == 0 or count == len(uids):
                print(f"contours={count}/{len(uids)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("uid")
    frame.to_csv(OUT / "contour_audit.csv", index=False)
    print(frame[["nonempty_contour_slices", "contour_voxels_total"]].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
