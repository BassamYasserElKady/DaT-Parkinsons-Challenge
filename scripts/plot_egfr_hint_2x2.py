#!/usr/bin/env python3
"""Plot a 2x2 calibrated-DeepSpot-M versus Xenium comparison for EGFR hints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


def correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--hints", type=Path, default=Path("results/egfr_tile_hints/development_hint_summary.csv"))
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/egfr_hint_2x2_calibrated_vs_xenium.png"))
    parser.add_argument("--metrics-output", type=Path, default=Path("results/egfr_hint_2x2_metrics.csv"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = pd.read_csv(args.manifest if args.manifest.is_absolute() else root / args.manifest)
    development = manifest.loc[manifest["split"] == "development", "sample_id"].astype(str).tolist()
    hint_table = pd.read_csv(args.hints if args.hints.is_absolute() else root / args.hints)
    hint_table = hint_table[hint_table["stable_positive_hint"]].sort_values("mean_spearman", ascending=False)
    genes = hint_table.head(args.top_n)["gene"].astype(str).tolist()
    if len(genes) != 4:
        raise ValueError(f"A 2x2 plot requires four stable positive hints; found {len(genes)}")

    checkpoint = torch.load(args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint, map_location="cpu", weights_only=False)
    model_genes = [str(g) for g in checkpoint["genes"]]
    with np.load(args.stats if args.stats.is_absolute() else root / args.stats, allow_pickle=False) as stats_file:
        prediction_mean = stats_file["prediction_mean"].astype(np.float32)
        prediction_std = stats_file["prediction_std"].astype(np.float32)
        stats_genes = [str(g) for g in stats_file["genes"].tolist()]
    if stats_genes != model_genes:
        raise ValueError("Adapter checkpoint and normalization genes do not match")
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()

    pooled = {gene: {"calibrated": [], "target": [], "egfr": []} for gene in genes}
    for sample_id in development:
        sample_dir = root / args.tables_dir / sample_id
        prediction_frame = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
        target_frame = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
        if not prediction_frame[["spot", "x", "y"]].equals(target_frame[["spot", "x", "y"]]):
            raise ValueError(f"Coordinates do not match for {sample_id}")
        prediction = prediction_frame[model_genes].to_numpy(np.float32)
        normalized = (prediction - prediction_mean) / prediction_std
        with torch.inference_mode():
            calibrated_normalized = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        calibrated = calibrated_normalized * target_std + target_mean
        for gene in genes:
            idx = model_genes.index(gene)
            pooled[gene]["calibrated"].append(calibrated[:, idx])
            pooled[gene]["target"].append(target_frame[gene].to_numpy(np.float32))
            pooled[gene]["egfr"].append(target_frame["EGFR"].to_numpy(np.float32))

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, gene in zip(axes.flat, genes):
        calibrated = np.concatenate(pooled[gene]["calibrated"])
        target = np.concatenate(pooled[gene]["target"])
        egfr = np.concatenate(pooled[gene]["egfr"])
        model_pearson, model_spearman = correlations(calibrated, target)
        xenium_pearson, xenium_spearman = correlations(target, egfr)
        rows.append(
            {
                "gene": gene,
                "tiles": len(target),
                "calibrated_vs_xenium_pearson": model_pearson,
                "calibrated_vs_xenium_spearman": model_spearman,
                "xenium_gene_vs_egfr_pearson": xenium_pearson,
                "xenium_gene_vs_egfr_spearman": xenium_spearman,
            }
        )
        rng = np.random.default_rng(abs(hash(gene)) % (2**32))
        selected = rng.choice(len(target), size=min(30000, len(target)), replace=False)
        ax.scatter(target[selected], calibrated[selected], s=1.2, alpha=0.12, color="#2b6cb0", linewidths=0)
        lo = min(float(np.quantile(target, 0.01)), float(np.quantile(calibrated, 0.01)))
        hi = max(float(np.quantile(target, 0.99)), float(np.quantile(calibrated, 0.99)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=0.8)
        ax.set_title(
            f"{gene}\nCalibrated vs Xenium: Pearson {model_pearson:.3f}; Spearman {model_spearman:.3f}\n"
            f"Xenium {gene} vs EGFR: Pearson {xenium_pearson:.3f}; Spearman {xenium_spearman:.3f}"
        )
        ax.set_xlabel("Actual Xenium log1p(CP10K)")
        ax.set_ylabel("Calibrated DeepSpot-M output")
        ax.grid(alpha=0.2)

    fig.suptitle("Top development-derived EGFR-associated genes: calibrated DeepSpot-M vs Xenium", fontsize=15)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics_output = args.metrics_output if args.metrics_output.is_absolute() else root / args.metrics_output
    metrics.to_csv(metrics_output, index=False)
    print(f"output={output}")
    print(f"metrics={metrics_output}")
    print(f"genes={','.join(genes)}; tiles={len(np.concatenate(pooled[genes[0]]['target'])):,}; device={device}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
