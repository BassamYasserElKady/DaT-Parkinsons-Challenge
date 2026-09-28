#!/usr/bin/env python3
"""Plot adaptive contours fitted to high-intensity striatal points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull


STRIATUM_LABELS = (5, 6, 16, 17)


def top_points(data: np.ndarray, roi: np.ndarray, percent: float) -> np.ndarray:
    values = data[roi]
    count = max(3, int(np.ceil(values.size * percent / 100.0)))
    flat_indices = np.flatnonzero(roi)
    selected = flat_indices[np.argpartition(values, -count)[-count:]]
    return np.column_stack(np.unravel_index(selected, roi.shape))


def hull_polygon(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return None
    hull = ConvexHull(points, qhull_options="QJ")
    return points[hull.vertices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-percent", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.nii.gz"))
    if args.count > len(paths):
        raise ValueError(f"Requested {args.count} scans, but only found {len(paths)}")
    atlas = np.asarray(nib.load(str(args.atlas)).dataobj)
    striatum = np.isin(atlas, STRIATUM_LABELS)
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(paths, size=args.count, replace=False), key=lambda p: p.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "selected_scans.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "selected_z_start", "selected_z_end", "selected_z_indices", "output_png"])
        writer.writeheader()
        for path in selected:
            uid = path.name.removesuffix(".nii.gz")
            data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
            data = np.clip(data, 0, None)
            slice_points = []
            slice_scores = []
            for z in range(data.shape[2]):
                roi = striatum[:, :, z]
                points = top_points(data[:, :, z], roi, args.top_percent)
                slice_points.append(points)
                slice_scores.append(float(data[points[:, 0], points[:, 1], z].sum()))
            _, start = max((sum(slice_scores[z : z + 7]), z) for z in range(data.shape[2] - 6))
            zs = list(range(start, start + 7))
            vmax = max(float(np.percentile(data[striatum], 99.5)), 1e-12)
            fig, axes = plt.subplots(1, 7, figsize=(21, 4), constrained_layout=True)
            for ax, z in zip(axes, zs):
                image = np.rot90(data[:, :, z])
                raw_points = slice_points[z]
                rotated_points = np.column_stack((raw_points[:, 0], data.shape[1] - 1 - raw_points[:, 1]))
                ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
                midpoint = image.shape[1] / 2
                for side_points in (rotated_points[rotated_points[:, 1] < midpoint], rotated_points[rotated_points[:, 1] >= midpoint]):
                    polygon = hull_polygon(side_points)
                    if polygon is not None:
                        ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="#ffe600", linewidth=1.8))
                ax.scatter(rotated_points[:, 0], rotated_points[:, 1], s=4, c="#ff5a36", alpha=0.65)
                ax.set_title(f"z={z}")
                ax.axis("off")
            fig.suptitle(f"{uid}: adaptive contours around top {args.top_percent:g}% striatal points")
            output_path = args.output_dir / f"{uid}_adaptive_striatum_contours.png"
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            writer.writerow({"uid": uid, "selected_z_start": start, "selected_z_end": start + 6, "selected_z_indices": ";".join(map(str, zs)), "output_png": str(output_path)})
            print(f"{uid}: z={start}-{start + 6} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
