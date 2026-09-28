#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "results/top20_intensity_matched_hotspot_composites"
MASK_PATH = ROOT / "assets/atlases/fsl/masks/striatal_voi_mask_LAS.nii.gz"
OUT = ROOT / "results/five_composite_striatal_overlays"
UIDS = ["bmae38sr", "7875aud9", "xid4n1u3", "y08p0fod", "uh17j8qo"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask_img = nib.load(str(MASK_PATH)); mask = mask_img.get_fdata() > 0
    z_counts = mask.sum(axis=(0, 1))
    active = np.flatnonzero(z_counts)
    center = int(np.round(np.average(active, weights=z_counts[active])))
    slices = list(range(max(int(active.min()), center - 4), min(int(active.max()) + 1, center + 4)))
    while len(slices) < 8: slices.append(slices[-1])
    slices = slices[:8]
    for uid in UIDS:
        path = COMP / f"{uid}_matched_composite.nii.gz"
        img = nib.load(str(path)); volume = img.get_fdata().astype(np.float32)
        if volume.shape != mask.shape or not np.allclose(img.affine, mask_img.affine):
            raise ValueError(f"Grid mismatch for {uid}")
        positive = volume[volume > 0]
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        for ax, z in zip(axes.flat, slices):
            base = np.rot90(volume[:, :, z].T)
            roi = np.rot90(mask[:, :, z].T)
            ax.imshow(base, cmap="gray", vmin=0, vmax=vmax)
            ax.contour(roi, levels=[0.5], colors="red", linewidths=1.5)
            ax.set_title(f"{uid} — axial z={z}")
            ax.axis("off")
        fig.suptitle(f"{uid}: intensity-matched composite with striatal VOI overlay", fontsize=14)
        fig.savefig(OUT / f"{uid}_composite_striatal_overlay.png", dpi=180)
        plt.close(fig)
    zip_path = ROOT / "results/five_composite_striatal_overlays.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.glob("*.png")):
            archive.write(path, path.name)
    print(f"scans={len(UIDS)} slices={slices}")
    print(zip_path)

if __name__ == "__main__":
    main()
