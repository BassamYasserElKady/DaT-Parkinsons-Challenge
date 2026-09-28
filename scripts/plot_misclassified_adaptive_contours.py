#!/usr/bin/env python3
"""Plot adaptive seven-slice contours for held-out model errors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import nibabel as nib
import numpy as np

from plot_random_adaptive_striatum_contours import hull_polygon, top_points


STRIATUM_LABELS = (5, 6, 16, 17)
LEFT_LABELS = (5, 6)
RIGHT_LABELS = (16, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--include-correct", action="store_true")
    return parser.parse_args()


def error_rows(predictions: Path, include_correct: bool) -> list[dict[str, str]]:
    with predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if include_correct or int(row["is_pathologic"]) != int(row["prediction"])]


def plot_scan(
    path: Path,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    top_percent: float,
    output_path: Path,
    title: str,
) -> tuple[int, int]:
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
    data = np.clip(data, 0, None)
    if data.shape != left_mask.shape or data.shape != right_mask.shape:
        raise ValueError(f"Shape mismatch for {path.name}: image {data.shape}, atlas {left_mask.shape}")

    striatum_mask = left_mask | right_mask
    slice_scores: list[float] = []
    for z in range(data.shape[2]):
        points = top_points(data[:, :, z], striatum_mask[:, :, z], top_percent)
        slice_scores.append(float(data[points[:, 0], points[:, 1], z].sum()))
    if data.shape[2] < 7:
        raise ValueError(f"{path.name} has fewer than seven slices")
    start = max(range(data.shape[2] - 6), key=lambda z: sum(slice_scores[z : z + 7]))
    zs = list(range(start, start + 7))

    points_by_side: dict[str, dict[int, np.ndarray]] = {}
    for name, side_mask in (("left", left_mask), ("right", right_mask)):
        window_mask = side_mask[:, :, zs]
        side_values = data[:, :, zs][window_mask]
        count = max(3, int(np.ceil(side_values.size * top_percent / 100.0)))
        flat_indices = np.flatnonzero(window_mask)
        selected_flat = flat_indices[np.argpartition(side_values, -count)[-count:]]
        selected = np.column_stack(np.unravel_index(selected_flat, window_mask.shape))
        points_by_side[name] = {z: selected[selected[:, 2] == z - start, :2] for z in zs}

    vmax = max(float(np.percentile(data[striatum_mask], 99.5)), 1e-12)
    fig, axes = plt.subplots(1, 7, figsize=(21, 4), constrained_layout=True)
    for ax, z in zip(axes, zs):
        image = np.rot90(data[:, :, z])
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        for name, color in (("left", "#42d9ff"), ("right", "#ff5a36")):
            raw_points = points_by_side[name][z]
            rotated_points = np.column_stack((raw_points[:, 0], data.shape[1] - 1 - raw_points[:, 1]))
            polygon = hull_polygon(rotated_points)
            if polygon is not None:
                ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=color, linewidth=1.8))
            if len(rotated_points):
                ax.scatter(rotated_points[:, 0], rotated_points[:, 1], s=4, c=color, alpha=0.75)
        ax.set_title(f"z={z}")
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return start, start + 6


def main() -> None:
    args = parse_args()
    atlas = np.asarray(nib.load(str(args.atlas)).dataobj)
    left_mask = np.isin(atlas, LEFT_LABELS)
    right_mask = np.isin(atlas, RIGHT_LABELS)
    rows = error_rows(args.predictions, args.include_correct)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "uid",
        "true_label",
        "prediction",
        "probability",
        "error_type",
        "selected_z_start",
        "selected_z_end",
        "output_png",
    ]
    summary_path = args.output_dir / ("correct_scans.csv" if args.include_correct else "misclassified_scans.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            uid = row["uid"]
            path = args.input_dir / f"{uid}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            true_label = int(row["is_pathologic"])
            prediction = int(row["prediction"])
            error_type = ("correct_patient" if true_label else "correct_healthy") if true_label == prediction else ("false_positive" if prediction == 1 else "false_negative")
            output_path = args.output_dir / f"{uid}_{error_type}_adaptive_contours.png"
            start, end = plot_scan(
                path,
                left_mask,
                right_mask,
                args.top_percent,
                output_path,
                f"{uid} | true={true_label} predicted={prediction} ({error_type}) | adaptive top {args.top_percent:g}%",
            )
            writer.writerow(
                {
                    "uid": uid,
                    "true_label": true_label,
                    "prediction": prediction,
                    "probability": row["probability"],
                    "error_type": error_type,
                    "selected_z_start": start,
                    "selected_z_end": end,
                    "output_png": str(output_path),
                }
            )
            print(f"{uid}: {error_type}, z={start}-{end} -> {output_path}", flush=True)
    print(f"Wrote {len(rows)} figures and {summary_path}")


if __name__ == "__main__":
    main()
