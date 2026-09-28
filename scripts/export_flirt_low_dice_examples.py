#!/usr/bin/env python3

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "fsl_flirt_affine12" / "dice_comparison" / "per_scan_dice.csv"
SCANS = ROOT / "results" / "fsl_flirt_affine12" / "registered"
TEMPLATE = ROOT / "submission_src_ensemble" / "assets" / "template.nii"
OUT = ROOT / "results" / "fsl_flirt_affine12" / "lowest_dice_10_pngs"
ZIP_PATH = ROOT / "results" / "fsl_flirt_affine12_lowest_dice_10_pngs.zip"


def normalize(volume):
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(positive, [1, 99])
    if high <= low:
        high = low + 1.0
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def views(volume, indices):
    x, y, z = indices
    return [np.rot90(volume[:, :, z]), np.rot90(volume[:, y, :]), np.rot90(volume[x, :, :])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(METRICS)
    selected = table[table["fsl_flirt_affine12_dice"] < 0.85].nsmallest(10, "fsl_flirt_affine12_dice")
    template = normalize(np.asarray(nib.load(str(TEMPLATE)).dataobj, dtype=np.float32))

    generated = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        scan = normalize(np.asarray(nib.load(str(SCANS / f"{row.uid}.nii.gz")).dataobj, dtype=np.float32))
        combined = template + scan
        center = np.unravel_index(int(np.argmax(combined)), combined.shape)
        template_views = views(template, center)
        scan_views = views(scan, center)

        fig, axes = plt.subplots(3, 3, figsize=(12, 11), facecolor="white")
        planes = ("Axial", "Coronal", "Sagittal")
        for column, plane in enumerate(planes):
            axes[0, column].imshow(template_views[column], cmap="gray", vmin=0, vmax=1)
            axes[0, column].set_title(f"Template — {plane}")
            axes[1, column].imshow(scan_views[column], cmap="gray", vmin=0, vmax=1)
            axes[1, column].set_title(f"FLIRT 12-DOF — {plane}")
            overlay = np.zeros((*template_views[column].shape, 3), dtype=np.float32)
            overlay[..., 0] = template_views[column]
            overlay[..., 1] = scan_views[column]
            overlay[..., 2] = template_views[column]
            axes[2, column].imshow(overlay)
            axes[2, column].set_title("Overlay: template=magenta, scan=green")
            for row_index in range(3):
                axes[row_index, column].axis("off")

        fig.suptitle(
            f"{row.uid} | FLIRT Dice={row.fsl_flirt_affine12_dice:.4f} | "
            f"ensemble ANTs Dice={row.ensemble_ants_rigid_dice:.4f}",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = OUT / f"{rank:02d}_{row.uid}_flirt_dice_{row.fsl_flirt_affine12_dice:.4f}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    with ZipFile(ZIP_PATH, "w", ZIP_DEFLATED) as archive:
        for path in generated:
            archive.write(path, arcname=path.name)
    print(f"generated={len(generated)}")
    print(f"zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
