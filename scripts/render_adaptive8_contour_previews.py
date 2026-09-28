#!/usr/bin/env python3
"""Render five representative adaptive-eight-slice contour stacks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "adaptive8_per_slice_contours"
REGISTERED = ROOT / "results" / "end_to_end_raw_registration_cv5_seed20260931" / "registered"
OUT = BASE / "preview_5_stacked_contours"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(BASE / "contour_audit.csv", dtype={"uid": str})
    # Representative complete stacks spanning the observed contour-volume range.
    complete = audit.loc[audit.nonempty_contour_slices.eq(8)].sort_values("contour_voxels_total")
    selected = complete.iloc[np.linspace(0, len(complete) - 1, 5, dtype=int)]
    for row in selected.itertuples(index=False):
        uid, start, end = row.uid, int(row.adaptive8_start), int(row.adaptive8_end)
        image = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        contour = np.asarray(nib.load(str(BASE / "contours" / f"{uid}_adaptive8_contours.nii.gz")).dataobj) > 0
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), facecolor="white")
        positive = image[image > 0]
        low, high = np.percentile(positive, [1, 99.5])
        for offset, axis in enumerate(axes.ravel()):
            z = start + offset
            # Transpose so displayed x/y agrees with the prior axial previews.
            axis.imshow(image[:, :, z].T, cmap="gray", origin="lower", vmin=low, vmax=high)
            mask = contour[:, :, z].T
            if mask.any():
                axis.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
            axis.set_title(f"z={z}  |  {int(mask.sum())} contour voxels", fontsize=10)
            axis.axis("off")
        fig.suptitle(f"{uid} — adaptive eight-slice independent contours (no MIP)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT / f"{uid}_adaptive8_contour_stack.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(uid, flush=True)


if __name__ == "__main__":
    main()
