#!/usr/bin/env python3
"""Visualize adaptive five-slice ROI crops with fold-specific masks."""
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
OUT = ADAPTIVE / "crop_level_visualizations"
SEED = 20260931
CROP_LEVELS = tuple(range(0, 100, 10))


def outline(mask):
    return mask & ~binary_erosion(mask)


def normalize(volume):
    positive = volume[np.isfinite(volume) & (volume > 0)]
    lo, hi = np.percentile(positive, [1, 99.5])
    return np.clip((volume - lo) / max(hi - lo, 1e-8), 0, 1)


def crop_bounds(mask, level, shape):
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    width = max(x1 - x0, y1 - y0)
    margin = int(round(width * level / 100.0))
    return max(0, x0 - margin), min(shape[0], x1 + margin), max(0, y0 - margin), min(shape[1], y1 + margin)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "results/cv5_submission_flip_clip/oof_predictions.csv", dtype={"uid": str})
    labels = labels.sort_values("uid").reset_index(drop=True)
    windows = pd.read_csv(ADAPTIVE / "adaptive_windows.csv", dtype={"uid": str}).set_index("uid")
    prior = pd.read_csv(ADAPTIVE / "top10_extreme_adaptive_shifts_manifest.csv", dtype={"uid": str})
    uids = prior.uid.astype(str).tolist()

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(labels, labels.actual))
    fold_map = {}
    for fold, (_, valid) in enumerate(splits, 1):
        for uid in labels.iloc[valid].uid.astype(str):
            fold_map[uid] = fold

    brain = np.asarray(nib.load(str(MASK_DIR / "MNI152_T1_2mm_brain_mask.nii.gz")).dataobj) > 0
    original = {name: np.asarray(nib.load(str(MASK_DIR / f"{name}.nii.gz")).dataobj) > 0 for name in ["left_caudate", "right_caudate", "left_putamen", "right_putamen"]}
    xcoord = np.indices(brain.shape)[0]
    fold_masks = {}
    for fold in range(1, 6):
        train, _ = splits[fold - 1]
        healthy = labels.iloc[train].loc[labels.iloc[train].actual == 0, "uid"].astype(str)
        total = np.zeros(brain.shape, dtype=np.float64)
        for uid in healthy:
            total += grid.scaled(uid, brain)
        template = (total / len(healthy)).astype(np.float32)
        base_ca, base_pu, _ = base.reposition(template, original)
        ca = grid.circularize(base_ca & (xcoord >= 45), 0.9) | grid.circularize(base_ca & (xcoord < 45), 0.9)
        pu = (grid.shorten(base_pu & (xcoord >= 45), 0.6) | grid.shorten(base_pu & (xcoord < 45), 0.6)) & ~ca
        fold_masks[fold] = (ca, pu)

    manifest = []
    for uid in uids:
        volume = np.asarray(nib.load(str(REG / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        start, end = int(windows.loc[uid, "start"]), int(windows.loc[uid, "end"])
        shift = int(start + 2 - 38)
        ca, pu = fold_masks[fold_map[uid]]
        ca = base.translate(ca, np.array([0, 0, shift], dtype=int))
        pu = base.translate(pu, np.array([0, 0, shift], dtype=int))
        slab = volume[:, :, start:end + 1].max(axis=2)
        ca2, pu2 = ca.max(axis=2), pu.max(axis=2)
        roi = ca2 | pu2
        scaled = normalize(slab)
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), facecolor="white")
        for ax, level in zip(axes.ravel(), CROP_LEVELS):
            x0, x1, y0, y1 = crop_bounds(roi, level, slab.shape)
            shown = scaled[x0:x1, y0:y1].T
            ax.imshow(shown, cmap="gray", origin="lower", vmin=0, vmax=1)
            ca_crop = outline(ca2[x0:x1, y0:y1])
            pu_crop = outline(pu2[x0:x1, y0:y1])
            ax.contour(ca_crop.T, levels=[0.5], colors=["cyan"], linewidths=1.3)
            ax.contour(pu_crop.T, levels=[0.5], colors=["lime"], linewidths=1.3)
            ax.set_title(f"context {level}%\n{x1-x0}×{y1-y0}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            manifest.append({"uid": uid, "crop_context_pct": level, "window_start": start, "window_end": end, "z_shift": shift, "x0": x0, "x1": x1, "y0": y0, "y1": y1})
        fig.suptitle(f"{uid} — adaptive five-slice MIP with shifted masks | window z={start}–{end} | shift={shift:+d}\ncyan=caudate, lime=putamen", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(OUT / f"{uid}_adaptive_crop_levels.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(manifest).to_csv(OUT / "crop_level_manifest.csv", index=False)
    print(f"created {len(uids)} PNGs in {OUT}")


if __name__ == "__main__":
    main()
