#!/usr/bin/env python3
"""Visualize MIP-only 2x2 background-block shuffles for Inception augmentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results" / "end_to_end_raw_registration_cv5_seed20260931" / "registered"
BRAIN = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
SRC = ROOT / "submission_ensemble_sigma075_8slice_svm" / "src"
OUT = ROOT / "results" / "inception_mip_background_block_shuffle_preview"


def make_mip_and_mask(uid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    sys.path.insert(0, str(SRC))
    from adaptive_localizer import adaptive_localization
    from svm_features import retain_top_bins, segment
    volume = np.asarray(nib.load(str(REG / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    best, _ = adaptive_localization(volume)
    center = (int(best["start"]) + int(best["end"])) // 2
    start = max(0, min(center - 4, volume.shape[2] - 8)); end = start + 8
    raw_mip = volume[:, :, start:end].max(axis=2)
    brain2 = brain[:, :, start:end].any(axis=2)
    striatum = segment(retain_top_bins(raw_mip, brain2), brain2)
    positive = raw_mip[brain2 & (raw_mip > 0)]
    normalized = np.zeros_like(raw_mip, dtype=np.float32)
    if len(positive): normalized[brain2] = (raw_mip[brain2] - positive.mean()) / max(positive.std(), 1e-6)
    return normalized, striatum, brain2, start, end - 1


def shuffle_blocks_to_random_contour_targets(image: np.ndarray, permitted: np.ndarray, contour_mean: float, low: float, high: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Shuffle blocks and independently set each block mean to a sampled range."""
    blocks = []
    for x in range(0, image.shape[0] - 1, 2):
        for y in range(0, image.shape[1] - 1, 2):
            if permitted[x:x+2, y:y+2].all(): blocks.append((x, y))
    output = image.copy()
    order = rng.permutation(len(blocks)); source = image.copy()
    targets = rng.uniform(low, high, size=len(blocks)) * contour_mean
    for destination_index, source_index in enumerate(order):
        dx, dy = blocks[destination_index]; sx, sy = blocks[source_index]
        block = source[sx:sx+2, sy:sy+2]
        source_mean = float(block.mean())
        if abs(source_mean) > 1e-6:
            output[dx:dx+2, dy:dy+2] = block * (targets[destination_index] / source_mean)
        else:
            output[dx:dx+2, dy:dy+2] = targets[destination_index]
    return output, len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--uid", default="085rntdr"); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    image, striatum, brain2, start, end = make_mip_and_mask(args.uid)
    # A valid block must be fully intracranial and fully outside the striatal contour.
    permitted = brain2 & (~striatum)
    variants = [("Original", image)]
    rng = np.random.default_rng(sum(map(ord, args.uid)) + 20260931)
    contour_mean = float(image[striatum].mean()) if striatum.any() else 0.0
    for low, high in ((.03, .70), (.10, .75), (.20, .80)):
        variant, n_blocks = shuffle_blocks_to_random_contour_targets(image, permitted, contour_mean, low, high, rng)
        variants.append((f"independent blocks: {low*100:.0f}%–{high*100:.0f}% contour mean", variant))
    vmin, vmax = np.percentile(image[np.isfinite(image)], [1, 99])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), facecolor="white")
    for axis, (title, variant) in zip(axes, variants):
        axis.imshow(variant.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axis.contour(striatum.T, levels=[.5], colors="red", linewidths=1.25)
        axis.set_title(title, fontsize=10); axis.axis("off")
    fig.suptitle(f"{args.uid} — adaptive 8-slice MIP z={start}–{end}; red = fixed striatal mask; {n_blocks} shuffled intracranial 2×2 blocks", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .90))
    path = OUT / f"{args.uid}_mip_block_shuffle_random_03_70_10_75_20_80.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); print(path)


if __name__ == "__main__": main()
