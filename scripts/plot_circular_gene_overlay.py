#!/usr/bin/env python3
"""Plot a circular H&E crop with raw, calibrated, and Xenium gene overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tifffile
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(finite, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    return (
        float(np.corrcoef(x[valid], y[valid])[0, 1]),
        float(spearmanr(x[valid], y[valid]).statistic),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-id", default="S4_Mid")
    parser.add_argument("--gene", default="ERBB2")
    parser.add_argument("--circle-fraction", type=float, default=0.45)
    parser.add_argument("--pyramid-level", type=int, default=4)
    parser.add_argument("--overlay-alpha", type=float, default=0.57)
    parser.add_argument("--marker-size", type=float, default=28.0)
    parser.add_argument("--output", type=Path, default=Path("results/s4_mid_erbb2_circular_overlay.png"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not 0.05 <= args.circle_fraction <= 0.5:
        raise ValueError("circle fraction must be between 0.05 and 0.5")
    if not 0.0 < args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be between 0 and 1")

    sample_dir = root / "results/deepspot_tables" / args.sample_id
    prediction_frame = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
    target_frame = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
    checkpoint = torch.load(root / "results/adapter_baseline/adapter_final.pt", map_location="cpu", weights_only=False)
    with np.load(root / "results/deepspot_tables/development_normalization.npz", allow_pickle=False) as stats:
        genes = [str(gene) for gene in stats["genes"].tolist()]
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
    if genes != [str(gene) for gene in checkpoint["genes"]]:
        raise ValueError("adapter checkpoint and development statistics genes do not match")
    if args.gene not in genes:
        raise ValueError(f"{args.gene} is not in the adapter gene list")
    if not prediction_frame[["spot", "x", "y"]].equals(target_frame[["spot", "x", "y"]]):
        raise ValueError("prediction and target coordinates do not match")

    prediction = prediction_frame[genes].to_numpy(np.float32)
    normalized_input = (prediction - prediction_mean) / prediction_std
    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()
    with torch.inference_mode():
        calibrated_normalized = model(torch.from_numpy(normalized_input).to(device)).cpu().numpy()
    calibrated = calibrated_normalized * checkpoint["target_std"] + checkpoint["target_mean"]

    manifest = pd.read_csv(root / "data/sample_manifest.csv")
    row = manifest.loc[manifest["sample_id"].astype(str) == args.sample_id]
    if len(row) != 1:
        raise ValueError(f"could not resolve one manifest row for {args.sample_id}")
    image_path = resolve(root, Path(str(row.iloc[0]["he_image_path"])))
    with tifffile.TiffFile(image_path) as tif:
        levels = tif.series[0].levels
        if args.pyramid_level >= len(levels):
            raise ValueError("requested H&E pyramid level is unavailable")
        level = levels[args.pyramid_level]
        image = level.asarray()
        full_height, full_width = levels[0].shape[:2]
    display_height, display_width = image.shape[:2]
    display_scale = ((full_width / display_width) + (full_height / display_height)) / 2.0
    image = image.copy()

    center_x, center_y = display_width / 2.0, display_height / 2.0
    radius = min(display_width, display_height) * args.circle_fraction
    yy, xx = np.ogrid[:display_height, :display_width]
    circle_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    image[~circle_mask] = 255

    tile_x = (prediction_frame["x"].to_numpy(np.float64) + 112.0) / display_scale
    tile_y = (prediction_frame["y"].to_numpy(np.float64) + 112.0) / display_scale
    in_circle = (tile_x - center_x) ** 2 + (tile_y - center_y) ** 2 <= radius ** 2
    tile_x, tile_y = tile_x[in_circle], tile_y[in_circle]
    gene_index = genes.index(args.gene)
    raw = prediction_frame.loc[in_circle, args.gene].to_numpy(np.float32)
    actual = target_frame.loc[in_circle, args.gene].to_numpy(np.float32)
    calibrated_gene = calibrated[in_circle, gene_index]

    raw_corr = corr_pair(raw, actual)
    calibrated_corr = corr_pair(calibrated_gene, actual)
    values = [raw, calibrated_gene, actual]
    labels = [
        f"Raw DeepSpot-M\nPearson {raw_corr[0]:.3f} | Spearman {raw_corr[1]:.3f}",
        f"Calibrated DeepSpot-M\nPearson {calibrated_corr[0]:.3f} | Spearman {calibrated_corr[1]:.3f}",
        "Actual Xenium\nreference",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, label, value in zip(axes, labels, values):
        low, high = robust_limits(value)
        ax.imshow(image, origin="upper")
        scatter = ax.scatter(tile_x, tile_y, c=value, cmap="magma", vmin=low, vmax=high,
                             s=args.marker_size, linewidths=0, alpha=args.overlay_alpha)
        circle = plt.Circle((center_x, center_y), radius, fill=False, color="cyan", linewidth=1.5)
        ax.add_patch(circle)
        ax.set_xlim(center_x - radius, center_x + radius)
        ax.set_ylim(center_y + radius, center_y - radius)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("H&E x (display pixels)")
        ax.set_ylabel("H&E y (display pixels)")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="expression")

    fig.suptitle(
        f"{args.sample_id} {args.gene} circular central-region comparison on H&E\n"
        f"{len(value):,} tiles inside circle; radius={args.circle_fraction:.2f} × minimum image dimension",
        fontsize=15,
    )
    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"output={output}")
    print(f"sample={args.sample_id}; gene={args.gene}; tiles_in_circle={len(raw):,}; device={device}")
    print(f"raw_pearson={raw_corr[0]:.6f}; raw_spearman={raw_corr[1]:.6f}")
    print(f"calibrated_pearson={calibrated_corr[0]:.6f}; calibrated_spearman={calibrated_corr[1]:.6f}")
    print(f"circle_center_display=({center_x:.1f}, {center_y:.1f}); radius_display={radius:.1f}")


if __name__ == "__main__":
    main()
