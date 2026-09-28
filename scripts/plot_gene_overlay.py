#!/usr/bin/env python3
"""Plot raw DeepSpot-M, calibrated, and Xenium gene overlays on H&E.

This script is visualization-only. It does not train, alter raw data, or write
anything except the requested output figure.
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
import tifffile
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-id", default="S1_Top")
    parser.add_argument("--gene", default="ERBB2")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/s1_top_erbb2_overlay.png"))
    parser.add_argument("--pyramid-level", type=int, default=4)
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def load_h_and_e(path: Path, level: int) -> tuple[np.ndarray, float]:
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        if level < 0 or level >= len(levels):
            raise ValueError(f"pyramid level {level} is unavailable; image has {len(levels)} levels")
        image = levels[level].asarray()
        full_height, full_width = levels[0].shape[:2]
        display_height, display_width = image.shape[:2]
    scale_x = full_width / display_width
    scale_y = full_height / display_height
    if not np.isclose(scale_x, scale_y, rtol=0.01):
        raise ValueError("H&E pyramid has unequal x/y downsampling")
    return image, float((scale_x + scale_y) / 2.0)


def robust_norm(values: np.ndarray) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin, vmax = np.quantile(finite, [0.01, 0.99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sample_dir = resolve(root, args.tables_dir) / args.sample_id
    prediction_path = next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet"))
    target_path = next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet"))
    prediction_frame = pd.read_parquet(prediction_path)
    target_frame = pd.read_parquet(target_path)

    checkpoint = torch.load(resolve(root, args.checkpoint), map_location="cpu", weights_only=False)
    genes = [str(gene) for gene in checkpoint["genes"]]
    if args.gene not in genes:
        raise ValueError(f"{args.gene} is not in the fitted adapter gene list")
    if args.gene not in prediction_frame or args.gene not in target_frame:
        raise ValueError(f"{args.gene} is missing from the processed tables")
    if not prediction_frame[["spot", "x", "y"]].equals(target_frame[["spot", "x", "y"]]):
        raise ValueError("prediction and target coordinates do not match")

    prediction = prediction_frame[genes].to_numpy(np.float32)
    target = target_frame[genes].to_numpy(np.float32)
    target_raw = target_frame[args.gene].to_numpy(np.float32)

    with np.load(resolve(root, args.stats), allow_pickle=False) as stats:
        stats_genes = [str(gene) for gene in stats["genes"].tolist()]
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
    if stats_genes != genes:
        raise ValueError("adapter checkpoint and development normalization genes do not match")
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    normalized_input = (prediction - prediction_mean) / prediction_std

    config = AdapterConfig(**checkpoint["adapter_config"])
    model = DeepSpotCalibrationAdapter(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()
    with torch.inference_mode():
        calibrated_normalized = model(torch.from_numpy(normalized_input).to(device)).cpu().numpy()
    calibrated = calibrated_normalized * target_std + target_mean
    calibrated_gene = calibrated[:, genes.index(args.gene)]
    raw_gene = prediction_frame[args.gene].to_numpy(np.float32)

    image_path = args.image
    if image_path is None:
        manifest = pd.read_csv(root / "data/sample_manifest.csv")
        row = manifest.loc[manifest["sample_id"].astype(str) == args.sample_id]
        if len(row) != 1:
            raise ValueError(f"could not resolve one manifest row for {args.sample_id}")
        image_path = Path(str(row.iloc[0]["he_image_path"]))
    image_path = resolve(root, image_path)
    image, display_scale = load_h_and_e(image_path, args.pyramid_level)

    # Tile coordinates are upper-left corners in full-resolution H&E pixels.
    x = prediction_frame["x"].to_numpy(np.float64) + 112.0
    y = prediction_frame["y"].to_numpy(np.float64) + 112.0
    x /= display_scale
    y /= display_scale

    values = [raw_gene, calibrated_gene, target_raw]
    titles = [
        "Raw DeepSpot-M",
        "Calibrated DeepSpot-M",
        "Actual Xenium",
    ]
    metrics = [
        (pearson(raw_gene, target_raw), float(spearmanr(raw_gene, target_raw).statistic)),
        (pearson(calibrated_gene, target_raw), float(spearmanr(calibrated_gene, target_raw).statistic)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, title, value in zip(axes, titles, values):
        ax.imshow(image, origin="upper")
        norm = robust_norm(value)
        scatter = ax.scatter(
            x,
            y,
            c=value,
            cmap="magma",
            norm=norm,
            s=5,
            linewidths=0,
            alpha=0.78,
        )
        ax.set_title(f"{title}\n{args.gene} log1p(CP10K)" if title == "Actual Xenium" else f"{title}\n{args.gene}")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_xlabel("H&E x (display pixels)")
        ax.set_ylabel("H&E y (display pixels)")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="expression")

    fig.suptitle(
        f"{args.sample_id} {args.gene} overlay on H&E | "
        f"raw Pearson={metrics[0][0]:.3f}, Spearman={metrics[0][1]:.3f}; "
        f"calibrated Pearson={metrics[1][0]:.3f}, Spearman={metrics[1][1]:.3f}\n"
        "Adapter fit on development samples; final-test sample shown for evaluation",
        fontsize=15,
    )
    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"output={output}")
    print(f"sample={args.sample_id}; gene={args.gene}; tiles={len(x):,}; device={device}")
    print(f"raw_spatial_pearson={metrics[0][0]:.6f}")
    print(f"raw_spatial_spearman={metrics[0][1]:.6f}")
    print(f"calibrated_spatial_pearson={metrics[1][0]:.6f}")
    print(f"calibrated_spatial_spearman={metrics[1][1]:.6f}")
    print(f"image_display_shape={image.shape}; pyramid_level={args.pyramid_level}; scale={display_scale:.6f}")


if __name__ == "__main__":
    main()
