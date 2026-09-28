#!/usr/bin/env python3
"""Create a six-panel, circular-cookie PGR/ERBB2 overlay figure.

Each row is a gene and each column is raw DeepSpotM, calibrated DeepSpotM,
or Xenium. Each panel contains only the circular cookie-cut region; pixels
outside the cookie are transparent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-id", default="S1_Top")
    parser.add_argument("--genes", nargs=2, default=["PGR", "ERBB2"])
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pyramid-level", type=int, default=4)
    parser.add_argument("--circle-fraction", type=float, default=0.27)
    parser.add_argument("--overlay-alpha", type=float, default=1.0)
    parser.add_argument("--he-alpha", type=float, default=0.78)
    return parser.parse_args()


def load_h_and_e(path: Path, level: int) -> tuple[np.ndarray, float]:
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        if level < 0 or level >= len(levels):
            raise ValueError(f"H&E pyramid level {level} is unavailable; image has {len(levels)} levels")
        image = levels[level].asarray()
        full_height, full_width = levels[0].shape[:2]
        display_height, display_width = image.shape[:2]
    scale_x = full_width / display_width
    scale_y = full_height / display_height
    if not np.isclose(scale_x, scale_y, rtol=0.01):
        raise ValueError("H&E pyramid has unequal x/y downsampling")
    return image, float((scale_x + scale_y) / 2.0)


def robust_norm(values: np.ndarray) -> Normalize:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    low, high = np.quantile(finite, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0
    return Normalize(vmin=float(low), vmax=float(high), clip=True)


def corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan"), float("nan")
    return (
        float(np.corrcoef(x[valid], y[valid])[0, 1]),
        float(spearmanr(x[valid], y[valid]).statistic),
    )


def fmt_corr(pair: tuple[float, float]) -> str:
    if not np.all(np.isfinite(pair)):
        return "r unavailable"
    return f"Pearson {pair[0]:.2f} | Spearman {pair[1]:.2f}"


def cookie_rgba(image: np.ndarray, center_x: float, center_y: float, radius: float, alpha: float = 1.0) -> np.ndarray:
    """Return the square crop as RGBA with transparent pixels outside the cookie."""
    height, width = image.shape[:2]
    left = max(0, int(np.floor(center_x - radius)))
    right = min(width, int(np.ceil(center_x + radius)))
    top = max(0, int(np.floor(center_y - radius)))
    bottom = min(height, int(np.ceil(center_y + radius)))
    crop = image[top:bottom, left:right]
    if crop.ndim == 2:
        crop = np.repeat(crop[..., None], 3, axis=2)
    if crop.shape[2] == 4:
        rgba = crop.copy()
    else:
        rgba = np.concatenate([crop[..., :3], np.full((*crop.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
    yy, xx = np.ogrid[top:bottom, left:right]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    rgba[..., 3] = np.where(mask, np.round(255.0 * alpha), 0).astype(np.uint8)
    return rgba


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if len(args.genes) != 2:
        raise ValueError("--genes must contain exactly two genes")
    if not 0.05 <= args.circle_fraction <= 0.5:
        raise ValueError("--circle-fraction must be between 0.05 and 0.5")
    if not 0.0 < args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")
    if not 0.0 < args.he_alpha <= 1.0:
        raise ValueError("--he-alpha must be between 0 and 1")

    sample_dir = resolve(root, args.tables_dir) / args.sample_id
    prediction_frame = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
    target_frame = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
    if not prediction_frame[["spot", "x", "y"]].equals(target_frame[["spot", "x", "y"]]):
        raise ValueError("prediction and target coordinates do not match")

    checkpoint = torch.load(resolve(root, args.checkpoint), map_location="cpu", weights_only=False)
    genes = [str(gene) for gene in checkpoint["genes"]]
    missing = [gene for gene in args.genes if gene not in genes or gene not in prediction_frame or gene not in target_frame]
    if missing:
        raise ValueError(f"genes missing from fitted tables/checkpoint: {missing}")

    with np.load(resolve(root, args.stats), allow_pickle=False) as stats:
        stats_genes = [str(gene) for gene in stats["genes"].tolist()]
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
    if stats_genes != genes:
        raise ValueError("adapter checkpoint and development normalization genes do not match")

    prediction = prediction_frame[genes].to_numpy(np.float32)
    normalized_input = (prediction - prediction_mean) / prediction_std
    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()
    with torch.inference_mode():
        calibrated_normalized = model(torch.from_numpy(normalized_input).to(device)).cpu().numpy()
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    calibrated = calibrated_normalized * target_std + target_mean

    manifest = pd.read_csv(root / "data/sample_manifest.csv")
    row = manifest.loc[manifest["sample_id"].astype(str) == args.sample_id]
    if len(row) != 1:
        raise ValueError(f"could not resolve one manifest row for {args.sample_id}")
    image_path = args.image
    if image_path is None:
        image_path = Path(str(row.iloc[0]["he_image_path"]))
    image, display_scale = load_h_and_e(resolve(root, image_path), args.pyramid_level)

    # Tile coordinates are upper-left corners in full-resolution H&E pixels.
    tile_x = (prediction_frame["x"].to_numpy(np.float64) + 112.0) / display_scale
    tile_y = (prediction_frame["y"].to_numpy(np.float64) + 112.0) / display_scale
    height, width = image.shape[:2]
    center_x = float(np.median(tile_x))
    center_y = float(np.median(tile_y))
    radius = float(min(width, height) * args.circle_fraction)
    circle_mask = (tile_x - center_x) ** 2 + (tile_y - center_y) ** 2 <= radius**2
    cookie = cookie_rgba(image, center_x, center_y, radius, alpha=args.he_alpha)

    modalities = ["Raw DeepSpotM", "Calibrated DeepSpotM", "Xenium"]
    # Every axis is a square crop whose outside pixels are transparent. The
    # only visible data are the circular cookie and its expression overlay.
    cookie_extent = (center_x - radius, center_x + radius, center_y + radius, center_y - radius)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    for row_index, gene in enumerate(args.genes):
        gene_index = genes.index(gene)
        raw = prediction_frame[gene].to_numpy(np.float32)
        cal = calibrated[:, gene_index]
        target = target_frame[gene].to_numpy(np.float32)
        values = [raw, cal, target]
        correlations = [corr_pair(raw, target), corr_pair(cal, target), None]

        # Raw DeepSpotM is in native model-output units. Calibrated and Xenium
        # share a target-space scale, which makes those two maps directly comparable.
        norms = [robust_norm(raw), robust_norm(np.concatenate([cal, target])), robust_norm(np.concatenate([cal, target]))]
        for column_index, (ax, modality, value, norm, correlation) in enumerate(
            zip(axes[row_index], modalities, values, norms, correlations)
        ):
            ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
            ax.imshow(cookie, origin="upper", extent=cookie_extent, interpolation="nearest")
            scatter = ax.scatter(
                tile_x[circle_mask],
                tile_y[circle_mask],
                c=value[circle_mask],
                cmap="magma",
                norm=norm,
                s=8.0,
                linewidths=0,
                alpha=args.overlay_alpha,
                rasterized=True,
            )
            ax.add_patch(plt.Circle((center_x, center_y), radius, fill=False, color="cyan", linewidth=1.4))
            unit = "native model scale" if modality.startswith("Raw") else "log1p(CP10K) scale"
            title = f"{gene} — {modality}\n{unit}"
            if correlation is not None:
                title += f"\n{fmt_corr(correlation)}"
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlim(center_x - radius, center_x + radius)
            ax.set_ylim(center_y + radius, center_y - radius)
            ax.set_aspect("equal")
            ax.axis("off")
            if column_index == 2:
                colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025, label="expression")
                colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "PGR and ERBB2 spatial comparison on H&E",
        fontsize=16,
        y=1.01,
    )
    output = resolve(root, args.output or Path(f"results/{args.sample_id.lower()}_pgr_erbb2_six_panel_cookie_overlay.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"output={output}")
    print(f"sample={args.sample_id}; genes={','.join(args.genes)}; tiles={len(tile_x):,}; device={device}")
    print(f"cookie_center_display=({center_x:.1f}, {center_y:.1f}); cookie_radius_display={radius:.1f}; tiles_in_cookie={int(circle_mask.sum()):,}")
    for gene in args.genes:
        i = genes.index(gene)
        raw = prediction_frame[gene].to_numpy(np.float32)
        cal = calibrated[:, i]
        target = target_frame[gene].to_numpy(np.float32)
        print(f"{gene}_raw_{fmt_corr(corr_pair(raw, target))}")
        print(f"{gene}_calibrated_{fmt_corr(corr_pair(cal, target))}")


if __name__ == "__main__":
    main()
