#!/usr/bin/env python3
"""Plot top-intensity striatal contours across the best contiguous seven slices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


STRIATUM_LABELS = (5, 6, 16, 17)  # bilateral caudate and putamen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-percent", type=float, default=1.0)
    return parser.parse_args()


def top_mask(values: np.ndarray, percent: float) -> np.ndarray:
    flat = values.reshape(-1)
    valid = np.flatnonzero(np.isfinite(flat))
    count = max(1, int(np.ceil(valid.size * percent / 100.0)))
    selected = valid[np.argpartition(flat[valid], -count)[-count:]]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(values.shape)


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.nii.gz"))
    if args.count > len(paths):
        raise ValueError(f"Requested {args.count} scans, but only found {len(paths)}")

    atlas_img = nib.load(str(args.atlas))
    atlas = np.asarray(atlas_img.dataobj)
    striatum = np.isin(atlas, STRIATUM_LABELS)
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(paths, size=args.count, replace=False), key=lambda p: p.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "selected_scans.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "selected_z_start", "selected_z_end", "selected_z_indices", "output_png"])
        writer.writeheader()
        for path in selected:
            uid = path.name.removesuffix(".nii.gz")
            data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
            if data.shape != striatum.shape:
                raise ValueError(f"{uid} shape {data.shape} does not match atlas {striatum.shape}")
            data = np.clip(data, 0, None)

            slice_masks = []
            slice_scores = []
            for z in range(data.shape[2]):
                mask = striatum[:, :, z]
                values = np.where(mask, data[:, :, z], np.nan)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    slice_masks.append(np.zeros(mask.shape, dtype=bool))
                    slice_scores.append(-np.inf)
                    continue
                high = top_mask(values, args.top_percent)
                slice_masks.append(high)
                slice_scores.append(float(values[high].sum()))

            windows = [(sum(slice_scores[z : z + 7]), z) for z in range(data.shape[2] - 6)]
            _, start = max(windows)
            zs = list(range(start, start + 7))
            vmax = float(np.percentile(data[striatum], 99.5))
            fig, axes = plt.subplots(1, 7, figsize=(21, 4), constrained_layout=True)
            for ax, z in zip(axes, zs):
                image = np.rot90(data[:, :, z])
                roi = np.rot90(striatum[:, :, z])
                high = np.rot90(slice_masks[z])
                ax.imshow(image, cmap="gray", vmin=0, vmax=max(vmax, 1e-12))
                ax.contour(roi.astype(float), levels=[0.5], colors="#00d9ff", linewidths=1.2)
                if high.any():
                    ax.contour(high.astype(float), levels=[0.5], colors="#ffe600", linewidths=1.4)
                ax.set_title(f"z={z}")
                ax.axis("off")
            fig.suptitle(f"{uid}: cyan=striatum, yellow=top {args.top_percent:g}% striatal voxels")
            output_path = args.output_dir / f"{uid}_striatum_contours.png"
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            writer.writerow({
                "uid": uid,
                "selected_z_start": start,
                "selected_z_end": start + 6,
                "selected_z_indices": ";".join(map(str, zs)),
                "output_png": str(output_path),
            })
            print(f"{uid}: z={start}-{start + 6} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
