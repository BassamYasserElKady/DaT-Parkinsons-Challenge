#!/usr/bin/env python3
"""Evaluate spatial correlations for genes highly expressed by model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def corr_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    if valid.sum() < 2 or np.std(prediction[valid]) == 0 or np.std(target[valid]) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(prediction[valid], target[valid])[0, 1])
    spearman = float(spearmanr(prediction[valid], target[valid]).statistic)
    return pearson, spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample-id", default="S2_Bot")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    sample_dir = resolve(root, args.tables_dir) / args.sample_id
    prediction_frame = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
    target_frame = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
    if not prediction_frame[["spot", "x", "y"]].equals(target_frame[["spot", "x", "y"]]):
        raise ValueError("Prediction and target tile coordinates do not match")

    checkpoint = torch.load(resolve(root, args.checkpoint), map_location="cpu", weights_only=False)
    genes = [str(gene) for gene in checkpoint["genes"]]
    if any(gene not in prediction_frame or gene not in target_frame for gene in genes):
        raise ValueError("The processed tables do not contain the complete fitted gene list")
    prediction = prediction_frame[genes].to_numpy(np.float32)
    target = target_frame[genes].to_numpy(np.float32)

    with np.load(resolve(root, args.stats), allow_pickle=False) as stats:
        stats_genes = [str(gene) for gene in stats["genes"].tolist()]
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
    if stats_genes != genes:
        raise ValueError("Checkpoint and development normalization genes do not match")

    sys.path.insert(0, str(root / "src"))
    from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device

    normalized_input = (prediction - prediction_mean) / prediction_std
    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()
    with torch.inference_mode():
        calibrated_normalized = model(torch.from_numpy(normalized_input).to(device)).cpu().numpy()
    calibrated = calibrated_normalized * np.asarray(checkpoint["target_std"], dtype=np.float32) + np.asarray(checkpoint["target_mean"], dtype=np.float32)

    raw_mean = prediction.mean(axis=0)
    calibrated_mean = calibrated.mean(axis=0)
    raw_order = np.argsort(-raw_mean)[: args.top_n]
    calibrated_order = np.argsort(-calibrated_mean)[: args.top_n]
    raw_set = set(raw_order.tolist())
    calibrated_set = set(calibrated_order.tolist())
    selected = sorted(raw_set | calibrated_set, key=lambda i: (-max(raw_mean[i], calibrated_mean[i]), genes[i]))

    rows = []
    for index in selected:
        raw_pearson, raw_spearman = corr_pair(prediction[:, index], target[:, index])
        calibrated_pearson, calibrated_spearman = corr_pair(calibrated[:, index], target[:, index])
        if index in raw_set and index in calibrated_set:
            selection_set = "both"
        elif index in raw_set:
            selection_set = "raw_only"
        else:
            selection_set = "calibrated_only"
        rows.append(
            {
                "sample_id": args.sample_id,
                "gene": genes[index],
                "selection_set": selection_set,
                "raw_rank_by_mean_expression": int(np.where(np.argsort(-raw_mean) == index)[0][0] + 1),
                "calibrated_rank_by_mean_expression": int(np.where(np.argsort(-calibrated_mean) == index)[0][0] + 1),
                "raw_mean_prediction": float(raw_mean[index]),
                "calibrated_mean_prediction": float(calibrated_mean[index]),
                "xenium_mean_target": float(target[:, index].mean()),
                "raw_spatial_pearson": raw_pearson,
                "raw_spatial_spearman": raw_spearman,
                "calibrated_spatial_pearson": calibrated_pearson,
                "calibrated_spatial_spearman": calibrated_spearman,
                "pearson_gain": calibrated_pearson - raw_pearson,
                "spearman_gain": calibrated_spearman - raw_spearman,
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["selection_set", "calibrated_rank_by_mean_expression", "gene"])
    output = args.output or Path(f"results/{args.sample_id.lower()}_high_expression_gene_metrics.csv")
    output = resolve(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)

    raw_genes = [genes[i] for i in raw_order]
    calibrated_genes = [genes[i] for i in calibrated_order]
    overlap = sorted(raw_set & calibrated_set, key=lambda i: genes[i])
    summary_path = output.with_suffix(".md")
    summary = [
        f"# High-expression spatial correlation evaluation: `{args.sample_id}`",
        "",
        f"Selection used the top `{args.top_n}` genes by mean raw DeepSpot-M prediction and the top `{args.top_n}` genes by mean calibrated prediction. Xenium values were not used for selection.",
        "",
        f"- Tiles evaluated: `{len(prediction):,}`",
        f"- Raw high-expression set: `{len(raw_genes)}` genes",
        f"- Calibrated high-expression set: `{len(calibrated_genes)}` genes",
        f"- Set overlap: `{len(overlap)}` genes",
        f"- Mean raw Pearson on selected union: `{metrics['raw_spatial_pearson'].mean():.4f}`",
        f"- Mean calibrated Pearson on selected union: `{metrics['calibrated_spatial_pearson'].mean():.4f}`",
        f"- Mean raw Spearman on selected union: `{metrics['raw_spatial_spearman'].mean():.4f}`",
        f"- Mean calibrated Spearman on selected union: `{metrics['calibrated_spatial_spearman'].mean():.4f}`",
        "",
        "## Raw top genes",
        "",
        ", ".join(raw_genes),
        "",
        "## Calibrated top genes",
        "",
        ", ".join(calibrated_genes),
        "",
        "The CSV contains gene-wise Pearson, Spearman, mean outputs, ranks, and calibration gains.",
    ]
    summary_path.write_text("\n".join(summary) + "\n")
    print(f"output={output}")
    print(f"summary={summary_path}")
    print(f"sample={args.sample_id}; tiles={len(prediction):,}; top_n={args.top_n}; union={len(selected)}; overlap={len(overlap)}; device={device}")
    print(f"mean_raw_pearson={metrics['raw_spatial_pearson'].mean():.6f}")
    print(f"mean_calibrated_pearson={metrics['calibrated_spatial_pearson'].mean():.6f}")
    print(f"mean_raw_spearman={metrics['raw_spatial_spearman'].mean():.6f}")
    print(f"mean_calibrated_spearman={metrics['calibrated_spatial_spearman'].mean():.6f}")


if __name__ == "__main__":
    main()
