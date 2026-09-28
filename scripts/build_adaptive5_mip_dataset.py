#!/usr/bin/env python3
"""Create one 2D axial MIP from each scan's adaptive five-slice window."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results/ants_registration_mni"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
LOCALIZER = ROOT / "submission_ensemble_sigma075_8slice_svm/src"
OUT = ROOT / "results/adaptive5_mip_2d"


def main() -> None:
    sys.path.insert(0, str(LOCALIZER))
    from adaptive_localizer import adaptive_localization

    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    mips = np.empty((len(rows), 109, 91), dtype=np.float32)
    audit = []

    for i, row in enumerate(rows):
        uid = row["uid"]
        volume = np.asarray(nib.load(str(REG / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        best, _ = adaptive_localization(volume)
        start, end = int(best["start"]), int(best["end"])
        if end - start + 1 != 5:
            raise RuntimeError(f"{uid}: localizer returned {start}:{end}")

        valid = np.isfinite(volume) & (volume > 0)
        positive = volume[valid]
        if positive.size == 0:
            raise RuntimeError(f"{uid}: empty positive volume")
        mean = float(positive.mean())
        std = float(positive.std())
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[valid] = (volume[valid] - mean) / max(std, 1e-8)

        # Match the existing array convention: transpose each axial slice to Y,X.
        mip = np.max(normalized[:, :, start:end + 1], axis=2).T
        if mip.shape != (109, 91):
            raise RuntimeError(f"{uid}: unexpected MIP shape {mip.shape}")
        mips[i] = mip
        audit.append({"uid": uid, "is_pathologic": row["is_pathologic"],
                      "start": start, "end": end,
                      "localizer_score": float(best["score"])})
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"processed={i + 1}/{len(rows)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "adaptive5_mips.npy", mips)
    with (OUT / "slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(f"saved={OUT / 'adaptive5_mips.npy'} shape={mips.shape}", flush=True)


if __name__ == "__main__":
    main()
