#!/usr/bin/env python3
from pathlib import Path
import csv
import zipfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ring_artifact_candidate_before_after_pngs"
RAW = ROOT / "data/external/niftis_original"
CLEAN = ROOT / "results/ants_registration_percentile10_masked/cleaned_masks"
REPORT = ROOT / "results/ants_registration_percentile10_masked/ring_artifact_candidates.csv"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(REPORT.open(newline="")))
    for row in rows:
        uid = row["uid"]
        image = np.asanyarray(nib.load(str(RAW / f"{uid}.nii.gz")).dataobj).astype(np.float32)
        mask = np.asanyarray(nib.load(str(CLEAN / f"{uid}.nii.gz")).dataobj) > 0
        scores = np.sum(np.maximum(image, 0), axis=(0, 1))
        z = int(np.argmax(scores))
        before = image[:, :, z]
        after = before * mask[:, :, z]
        vmax = float(np.percentile(before[before > 0], 99.5)) if np.any(before > 0) else 1.0
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        ax[0].imshow(np.rot90(before), cmap="gray", vmin=0, vmax=vmax)
        ax[0].set_title(f"Before cleanup\naxial slice {z}")
        ax[1].imshow(np.rot90(after), cmap="gray", vmin=0, vmax=vmax)
        ax[1].contour(np.rot90(mask[:, :, z].astype(float)), levels=[0.5], colors="red", linewidths=0.5)
        ax[1].set_title("After LCC + hole filling")
        for a in ax: a.axis("off")
        fig.suptitle(f"{uid} | radial-band score {float(row['ring_score']):.3f}")
        fig.savefig(OUT / f"{uid}.png", dpi=160)
        plt.close(fig)
    zip_path = ROOT / "results/ring_artifact_candidate_before_after_pngs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob("*.png")): z.write(p, p.name)
    print(f"pngs={len(list(OUT.glob('*.png')))}")
    print(f"zip={zip_path}")

if __name__ == "__main__": main()
