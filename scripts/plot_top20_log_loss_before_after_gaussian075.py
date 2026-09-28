#!/usr/bin/env python3
"""Render the 20 worst OOF log-loss scans before/after 3D Gaussian smoothing."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "results/cv5_real8_post_registration_gaussian"
DEFAULT_REGISTERED = ROOT / "results/ants_registration_mni"
SIGMA = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--registered", type=Path, default=DEFAULT_REGISTERED)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def per_scan_log_loss(actual: int, probability: float) -> float:
    probability = float(np.clip(probability, 1e-15, 1.0 - 1e-15))
    return float(-math.log(probability if actual else 1.0 - probability))


def read_ranked_rows(path: Path, count: int) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["actual"] = int(row["actual"])
        row["fold"] = int(row["fold"])
        row["equal_ensemble"] = float(row["equal_ensemble"])
        row["equal_ensemble_clipped"] = float(row["equal_ensemble_clipped"])
        row["per_scan_log_loss"] = per_scan_log_loss(row["actual"], row["equal_ensemble_clipped"])
    rows.sort(key=lambda row: (-float(row["per_scan_log_loss"]), str(row["uid"])))
    return rows[:count]


def load_slice_selection(path: Path) -> dict[str, dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["uid"]: {key: int(row[key]) for key in ("start", "end", "center")}
            for row in csv.DictReader(handle)
        }


def mip_for_window(volume: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return a display-oriented MIP matching the model's transposed slice layout."""
    return np.max(volume[:, :, start : end + 1], axis=2).T


def display_limits(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([before.ravel(), after.ravel()])
    combined = combined[np.isfinite(combined)]
    combined = np.log1p(np.maximum(combined, 0.0))
    low, high = np.percentile(combined, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(combined)), float(np.max(combined))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    registered = args.registered.resolve()
    output = (args.output or experiment / "top20_log_loss_before_after_sigma0p75.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    ranked = read_ranked_rows(experiment / "oof_predictions.csv", args.count)
    slices = load_slice_selection(experiment / "slice_selection.csv")

    # Two panels per scan: raw registered volume and the same volume after the
    # exact 3D sigma=0.75 voxel smoothing used by the model pipeline. The
    # already-selected 8-slice window is held fixed for the before/after pair.
    figure, axes = plt.subplots(
        args.count,
        2,
        figsize=(10.5, max(4.0, args.count * 2.05)),
        squeeze=False,
        constrained_layout=False,
    )
    figure.patch.set_facecolor("white")

    manifest_rows: list[dict[str, object]] = []
    for index, row in enumerate(ranked):
        uid = str(row["uid"])
        selection = slices[uid]
        volume = np.asarray(nib.load(str(registered / f"{uid}.nii.gz")).get_fdata(), dtype=np.float32)
        volume = np.where(np.isfinite(volume), np.maximum(volume, 0.0), 0.0)
        smoothed = ndimage.gaussian_filter(volume, sigma=SIGMA, mode="nearest")
        before = mip_for_window(volume, selection["start"], selection["end"])
        after = mip_for_window(smoothed, selection["start"], selection["end"])
        before_log = np.log1p(before)
        after_log = np.log1p(after)
        vmin, vmax = display_limits(before, after)

        title = (
            f"{index + 1:02d}. {uid}  |  y={row['actual']}  |  "
            f"p={float(row['equal_ensemble_clipped']):.3f}  |  "
            f"log loss={float(row['per_scan_log_loss']):.3f}  |  "
            f"z={selection['start'] + 1}–{selection['end'] + 1}"
        )
        axes[index, 0].imshow(before_log, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[index, 1].imshow(after_log, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[index, 0].set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8.5, labelpad=8)
        for column in range(2):
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
            axes[index, column].set_aspect("equal")

        manifest_rows.append(
            {
                "rank": index + 1,
                "uid": uid,
                "actual": row["actual"],
                "fold": row["fold"],
                "probability": row["equal_ensemble_clipped"],
                "per_scan_log_loss": row["per_scan_log_loss"],
                "slice_start_zero_based": selection["start"],
                "slice_end_zero_based": selection["end"],
                "slice_center_zero_based": selection["center"],
            }
        )

    axes[0, 0].set_title("Before smoothing", fontsize=11, pad=8)
    axes[0, 1].set_title("After 3D Gaussian smoothing (σ=0.75 voxels)", fontsize=11, pad=8)
    figure.suptitle(
        "Top 20 scans by per-scan OOF log loss\n"
        "MIP of the same 8-slice 3D model window; log-scaled intensity for display",
        fontsize=13,
        y=0.998,
    )
    figure.subplots_adjust(left=0.30, right=0.99, top=0.985, bottom=0.01, hspace=0.20, wspace=0.04)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest_path = output.with_suffix(".csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"output={output}")
    print(f"manifest={manifest_path}")
    print(f"scans={len(manifest_rows)} sigma_voxels={SIGMA}")


if __name__ == "__main__":
    main()
