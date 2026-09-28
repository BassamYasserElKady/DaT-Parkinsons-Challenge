#!/usr/bin/env python3
"""Plot adapter parity using one mean point per selected gene."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def main() -> None:
    manifest = pd.read_csv(resolve(Path("data/sample_manifest.csv")))
    final_test = manifest.loc[manifest["split"] == "final_test", "sample_id"].astype(str).tolist()

    stats_path = resolve(Path("results/deepspot_tables/development_normalization.npz"))
    with np.load(stats_path, allow_pickle=False) as data:
        genes = [str(gene) for gene in data["genes"].tolist()]

    selected_path = resolve(Path("results/adapter_evaluation/top_200_adapter_improved_genes.csv"))
    selected = pd.read_csv(selected_path)
    selected_genes = selected["gene"].astype(str).tolist()
    indices = [genes.index(gene) for gene in selected_genes]

    truth_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    adapter_parts: list[np.ndarray] = []
    for sample_id in final_test:
        sample_dir = resolve(Path("results/deepspot_tables")) / sample_id
        raw = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
        target = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
        coordinates = ["spot", "x", "y"]
        if not raw[coordinates].equals(target[coordinates]):
            raise ValueError(f"Prediction/target coordinates do not match for {sample_id}")
        adapter = np.load(resolve(Path("results/adapter_baseline")) / f"{sample_id}.npy")
        truth_parts.append(target[genes].to_numpy(np.float64)[:, indices])
        raw_parts.append(raw[genes].to_numpy(np.float64)[:, indices])
        adapter_parts.append(adapter.astype(np.float64)[:, indices])

    truth = np.concatenate(truth_parts, axis=0)
    raw_prediction = np.concatenate(raw_parts, axis=0)
    adapter_prediction = np.concatenate(adapter_parts, axis=0)

    raw_error = raw_prediction - truth
    adapter_error = adapter_prediction - truth
    gene_metrics = pd.DataFrame(
        {
            "selection_rank": selected["selection_rank"].to_numpy(int),
            "gene": selected_genes,
            "xenium_mean": truth.mean(axis=0),
            "raw_prediction_mean": raw_prediction.mean(axis=0),
            "adapter_prediction_mean": adapter_prediction.mean(axis=0),
            "raw_mean_error": raw_error.mean(axis=0),
            "adapter_mean_error": adapter_error.mean(axis=0),
            "raw_mae_across_tiles": np.mean(np.abs(raw_error), axis=0),
            "adapter_mae_across_tiles": np.mean(np.abs(adapter_error), axis=0),
            "raw_rmse_across_tiles": np.sqrt(np.mean(raw_error**2, axis=0)),
            "adapter_rmse_across_tiles": np.sqrt(np.mean(adapter_error**2, axis=0)),
            "pearson_gain": selected["pearson_gain"].to_numpy(float),
            "spearman_gain": selected["spearman_gain"].to_numpy(float),
        }
    )
    gene_metrics.to_csv(
        resolve(Path("results/adapter_evaluation/top_200_adapter_gene_mean_metrics.csv")),
        index=False,
    )

    truth_mean = gene_metrics["xenium_mean"].to_numpy()
    raw_mean = gene_metrics["raw_prediction_mean"].to_numpy()
    adapter_mean = gene_metrics["adapter_prediction_mean"].to_numpy()
    raw_mean_pearson, raw_mean_spearman = correlation(truth_mean, raw_mean)
    adapter_mean_pearson, adapter_mean_spearman = correlation(truth_mean, adapter_mean)

    combined = np.concatenate([truth_mean, raw_mean, adapter_mean])
    lo = min(0.0, float(np.nanmin(combined)))
    hi = max(1.0, float(np.nanmax(combined)))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    panels = [
        (axes[0], raw_mean, "Raw DeepSpotM", raw_mean_pearson, raw_mean_spearman),
        (axes[1], adapter_mean, "Adapter", adapter_mean_pearson, adapter_mean_spearman),
    ]
    for ax, prediction, title, pearson, spearman in panels:
        ax.scatter(truth_mean, prediction, s=24, alpha=0.78, color="#3182bd", edgecolors="white", linewidths=0.3)
        ax.plot([lo, hi], [lo, hi], color="#b2182b", linestyle="--", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Mean Xenium gene expression log1p(CP10K)")
        ax.set_ylabel("Mean predicted gene expression log1p(CP10K)")
        ax.set_title(title)
        ax.text(
            0.04, 0.96,
            f"N = {len(gene_metrics)} genes\n"
            f"Pearson r = {pearson:.3f}\n"
            f"Spearman ρ = {spearman:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
        )

    figure_path = resolve(Path("results/adapter_evaluation/top_200_adapter_gene_mean_parity_and_improvement.png"))
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"saved={figure_path}")
    print(f"saved={resolve(Path('results/adapter_evaluation/top_200_adapter_gene_mean_metrics.csv'))}")
    print(
        f"raw_mean_correlations=pearson:{raw_mean_pearson:.4f},spearman:{raw_mean_spearman:.4f}; "
        f"adapter_mean_correlations=pearson:{adapter_mean_pearson:.4f},spearman:{adapter_mean_spearman:.4f}"
    )


if __name__ == "__main__":
    main()
