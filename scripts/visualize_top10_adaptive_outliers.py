#!/usr/bin/env python3
"""One-page QC of the ten largest adaptive axial shifts."""
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from sklearn.model_selection import StratifiedKFold

import grid_mask_shape_sbr_cv5 as grid
import nested_repositioned_mask0_sbr_cv5 as base


ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results/ants_registration_mni"
MASK_DIR = ROOT / "assets/atlases/fsl/masks"
ADAPTIVE = ROOT / "results/adaptive_slice_best_mask_sbr_cv5"
OUTPUT = ADAPTIVE / "top10_extreme_adaptive_shifts_mask_and_top025.png"
SEED = 20260931


def outline(mask2d):
    return mask2d & ~binary_erosion(mask2d)


def normalized(image):
    finite = image[np.isfinite(image) & (image > 0)]
    if finite.size == 0:
        return np.zeros_like(image)
    lo, hi = np.percentile(finite, [1, 99.5])
    return np.clip((image - lo) / max(hi - lo, 1e-8), 0, 1)


def main():
    labels = pd.read_csv(
        ROOT / "results/cv5_submission_flip_clip/oof_predictions.csv", dtype={"uid": str}
    ).sort_values("uid").reset_index(drop=True)
    windows = pd.read_csv(ADAPTIVE / "adaptive_windows.csv", dtype={"uid": str}).set_index("uid")
    fold_map = {}
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(labels, labels.actual))
    for fold, (_, valid) in enumerate(splits, 1):
        for uid in labels.iloc[valid].uid.astype(str):
            fold_map[uid] = fold

    # All fold-specific healthy templates localized to z=36..40, center z=38.
    shifts = windows.start.astype(int) + 2 - 38
    top_uids = shifts.abs().sort_values(ascending=False).head(10).index.tolist()

    brain = np.asarray(nib.load(str(MASK_DIR / "MNI152_T1_2mm_brain_mask.nii.gz")).dataobj) > 0
    original = {
        name: np.asarray(nib.load(str(MASK_DIR / f"{name}.nii.gz")).dataobj) > 0
        for name in ["left_caudate", "right_caudate", "left_putamen", "right_putamen"]
    }
    xcoord = np.indices(brain.shape)[0]

    fold_masks = {}
    for fold in sorted({fold_map[uid] for uid in top_uids}):
        train, _ = splits[fold - 1]
        healthy = labels.iloc[train].loc[labels.iloc[train].actual == 0, "uid"].astype(str)
        total = np.zeros(brain.shape, dtype=np.float64)
        for uid in healthy:
            total += grid.scaled(uid, brain)
        template = (total / len(healthy)).astype(np.float32)
        base_ca, base_pu, _ = base.reposition(template, original)
        ca = (grid.circularize(base_ca & (xcoord >= 45), 0.9) |
              grid.circularize(base_ca & (xcoord < 45), 0.9))
        pu = (grid.shorten(base_pu & (xcoord >= 45), 0.6) |
              grid.shorten(base_pu & (xcoord < 45), 0.6)) & ~ca
        fold_masks[fold] = (ca, pu)

    fig, axes = plt.subplots(2, 10, figsize=(25, 6.2), facecolor="white")
    manifest = []
    for col, uid in enumerate(top_uids):
        volume = np.asarray(nib.load(str(REG / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        row = windows.loc[uid]
        start, end = int(row.start), int(row.end)
        shift = int(shifts.loc[uid])
        fold = fold_map[uid]
        ca, pu = fold_masks[fold]
        ca = base.translate(ca, np.array([0, 0, shift], dtype=int))
        pu = base.translate(pu, np.array([0, 0, shift], dtype=int))

        slab = volume[:, :, start:end + 1].max(axis=2)
        ca_proj = ca.max(axis=2)
        pu_proj = pu.max(axis=2)
        ax = axes[0, col]
        ax.imshow(normalized(slab).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.contour(outline(ca_proj).T, levels=[0.5], colors=["cyan"], linewidths=1.1)
        ax.contour(outline(pu_proj).T, levels=[0.5], colors=["lime"], linewidths=1.1)
        ax.set_title(f"{uid}\nz={start}–{end}, shift={shift:+d}", fontsize=9)

        positive = volume[np.isfinite(volume) & (volume > 0)]
        cutoff = float(np.percentile(positive, 99.75))
        hot = np.isfinite(volume) & (volume >= cutoff) & (volume > 0)
        hot_mip = np.where(hot, volume, 0).max(axis=2)
        ax = axes[1, col]
        ax.imshow(hot_mip.T, cmap="inferno", origin="lower")
        ax.set_title(f"top 0.25%\nn={int(hot.sum())}", fontsize=9)
        manifest.append({
            "rank": col + 1, "uid": uid, "fold": fold,
            "window_start": start, "window_end": end, "z_shift": shift,
            "top025_cutoff": cutoff, "top025_voxels": int(hot.sum()),
        })

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Ten largest adaptive axial-shift outliers — fold-specific shifted ROI and brightest 0.25% voxels\n"
        "cyan = caudate, lime = putamen; upper images are maximum projections of each selected five-slice window",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=0.35)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(manifest).to_csv(ADAPTIVE / "top10_extreme_adaptive_shifts_manifest.csv", index=False)
    print(OUTPUT)
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
