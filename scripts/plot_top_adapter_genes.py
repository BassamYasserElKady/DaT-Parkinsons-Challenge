#!/usr/bin/env python3
"""Plot held-out parity and error improvement for adapter-improved genes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("results/adapter_baseline"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adapter_evaluation"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def corr_columns(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson and Spearman correlation for each gene column."""
    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(truth_centered**2, axis=0) * np.sum(prediction_centered**2, axis=0)
    )
    pearson = np.divide(
        np.sum(truth_centered * prediction_centered, axis=0),
        denominator,
        out=np.full(truth.shape[1], np.nan),
        where=denominator > 0,
    )

    truth_rank = np.apply_along_axis(rankdata, 0, truth)
    prediction_rank = np.apply_along_axis(rankdata, 0, prediction)
    truth_rank -= truth_rank.mean(axis=0, keepdims=True)
    prediction_rank -= prediction_rank.mean(axis=0, keepdims=True)
    rank_denominator = np.sqrt(
        np.sum(truth_rank**2, axis=0) * np.sum(prediction_rank**2, axis=0)
    )
    spearman = np.divide(
        np.sum(truth_rank * prediction_rank, axis=0),
        rank_denominator,
        out=np.full(truth.shape[1], np.nan),
        where=rank_denominator > 0,
    )
    return pearson, spearman


def pooled_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {
        "n_values": int(truth.size),
        "pearson": float(np.corrcoef(truth, prediction)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(truth), rankdata(prediction))[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def add_stats(ax, title: str, values: dict[str, float]) -> None:
    ax.text(
        0.04, 0.96,
        f"{title}\n"
        f"N = {values['n_values']:,}\n"
        f"Pearson r = {values['pearson']:.3f}\n"
        f"Spearman ρ = {values['spearman']:.3f}\n"
        f"MAE = {values['mae']:.3f}\n"
        f"RMSE = {values['rmse']:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("top-n must be positive")

    manifest = pd.read_csv(resolve(args.manifest))
    final_test = manifest.loc[manifest["split"] == "final_test", "sample_id"].astype(str).tolist()
    with np.load(resolve(args.stats), allow_pickle=False) as data:
        genes = [str(gene) for gene in data["genes"].tolist()]

    truth_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    adapter_parts: list[np.ndarray] = []
    for sample_id in final_test:
        sample_dir = resolve(args.tables_dir) / sample_id
        raw = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
        target = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
        coordinates = ["spot", "x", "y"]
        if not raw[coordinates].equals(target[coordinates]):
            raise ValueError(f"Prediction/target coordinates do not match for {sample_id}")
        adapter = np.load(resolve(args.adapter_dir) / f"{sample_id}.npy")
        truth_parts.append(target[genes].to_numpy(np.float64))
        raw_parts.append(raw[genes].to_numpy(np.float64))
        adapter_parts.append(adapter.astype(np.float64))

    truth = np.concatenate(truth_parts, axis=0)
    raw_prediction = np.concatenate(raw_parts, axis=0)
    adapter_prediction = np.concatenate(adapter_parts, axis=0)
    if truth.shape != raw_prediction.shape or truth.shape != adapter_prediction.shape:
        raise ValueError("Truth, raw prediction, and adapter prediction shapes do not match")

    raw_pearson, raw_spearman = corr_columns(truth, raw_prediction)
    adapter_pearson, adapter_spearman = corr_columns(truth, adapter_prediction)
    gene_metrics = pd.DataFrame(
        {
            "gene": genes,
            "raw_pearson": raw_pearson,
            "adapter_pearson": adapter_pearson,
            "pearson_gain": adapter_pearson - raw_pearson,
            "raw_spearman": raw_spearman,
            "adapter_spearman": adapter_spearman,
            "spearman_gain": adapter_spearman - raw_spearman,
        }
    )
    gene_metrics["combined_gain"] = (gene_metrics["pearson_gain"] + gene_metrics["spearman_gain"]) / 2
    eligible = gene_metrics["pearson_gain"].gt(0) & gene_metrics["spearman_gain"].gt(0)
    selected = gene_metrics.loc[eligible].sort_values("combined_gain", ascending=False).head(args.top_n).copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected = selected[["selection_rank", "gene", "raw_pearson", "adapter_pearson", "pearson_gain", "raw_spearman", "adapter_spearman", "spearman_gain", "combined_gain"]]

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / f"top_{args.top_n}_adapter_improved_genes.csv", index=False)
    gene_metrics.sort_values("combined_gain", ascending=False).to_csv(
        output_dir / "all_gene_adapter_gain_metrics.csv", index=False
    )

    indices = [genes.index(gene) for gene in selected["gene"]]
    selected_truth = truth[:, indices].reshape(-1)
    selected_raw = raw_prediction[:, indices].reshape(-1)
    selected_adapter = adapter_prediction[:, indices].reshape(-1)
    raw_stats = pooled_metrics(selected_truth, selected_raw)
    adapter_stats = pooled_metrics(selected_truth, selected_adapter)
    pd.DataFrame(
        [
            {"model": "raw_deepspotm", **raw_stats},
            {"model": "adapter", **adapter_stats},
            {"model": "selection", "n_values": selected_truth.size, "n_genes": len(selected)},
        ]
    ).to_csv(output_dir / f"top_{args.top_n}_adapter_gene_plot_metrics.csv", index=False)

    combined = np.concatenate([selected_truth, selected_raw, selected_adapter])
    lo = min(0.0, float(np.nanpercentile(combined, 0.5)))
    hi = max(1.0, float(np.nanpercentile(combined, 99.5)))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    for ax, prediction, title, stats in [
        (axes[0], selected_raw, "Raw DeepSpotM", raw_stats),
        (axes[1], selected_adapter, "Adapter", adapter_stats),
    ]:
        ax.hexbin(selected_truth, prediction, gridsize=75, mincnt=1, bins="log", cmap="Blues", linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="#b2182b", linestyle="--", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Xenium gene expression log1p(CP10K)")
        ax.set_ylabel("Predicted gene expression log1p(CP10K)")
        ax.set_title(title)
        add_stats(ax, f"Top {len(selected)} genes", stats)

    raw_abs_error = np.abs(selected_raw - selected_truth)
    adapter_abs_error = np.abs(selected_adapter - selected_truth)
    error_hi = max(1.0, float(np.nanpercentile(np.concatenate([raw_abs_error, adapter_abs_error]), 99.5)))
    error_ax = axes[2]
    error_ax.hexbin(
        np.log1p(raw_abs_error), np.log1p(adapter_abs_error),
        gridsize=75, mincnt=1, bins="log", cmap="Purples", linewidths=0,
    )
    error_ax.plot([0, np.log1p(error_hi)], [0, np.log1p(error_hi)], color="#b2182b", linestyle="--", linewidth=1.2)
    error_ax.set_xlim(0, np.log1p(error_hi))
    error_ax.set_ylim(0, np.log1p(error_hi))
    error_ax.set_xlabel("Raw DeepSpotM absolute error, log1p")
    error_ax.set_ylabel("Adapter absolute error, log1p")
    error_ax.set_title("Per-value error: adapter vs raw")
    error_ax.text(
        0.04, 0.96,
        "Below diagonal = adapter is better\n"
        f"MAE: {raw_stats['mae']:.3f} → {adapter_stats['mae']:.3f}\n"
        f"RMSE: {raw_stats['rmse']:.3f} → {adapter_stats['rmse']:.3f}",
        transform=error_ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    fig.suptitle(
        f"Held-out adapter evaluation — top {len(selected)} genes improved in Pearson and Spearman",
        fontsize=14, fontweight="bold",
    )
    figure_path = output_dir / f"top_{args.top_n}_adapter_parity_and_improvement.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"eligible_genes={int(eligible.sum())}; selected_genes={len(selected)}")
    print(f"saved={figure_path}")
    print(f"saved={output_dir / f'top_{args.top_n}_adapter_improved_genes.csv'}")
    print(pd.DataFrame([raw_stats, adapter_stats], index=["raw", "adapter"]).to_string())


if __name__ == "__main__":
    main()
