#!/usr/bin/env python3
"""Plot held-out adapter accuracy and per-tile error improvement for one gene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="EGFR")
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("results/adapter_baseline"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adapter_evaluation"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y_true
    return {
        "n_tiles": int(len(y_true)),
        "pearson": float(np.corrcoef(y_true, prediction)[0, 1]),
        "spearman": float(spearmanr(y_true, prediction).statistic),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def add_stats(ax, label: str, values: dict[str, float], loc: tuple[float, float]) -> None:
    text = (
        f"{label}\n"
        f"N = {values['n_tiles']:,}\n"
        f"Pearson r = {values['pearson']:.3f}\n"
        f"Spearman ρ = {values['spearman']:.3f}\n"
        f"MAE = {values['mae']:.3f}\n"
        f"RMSE = {values['rmse']:.3f}"
    )
    ax.text(
        loc[0], loc[1], text, transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(resolve(args.manifest))
    final_test = manifest.loc[manifest["split"] == "final_test", "sample_id"].astype(str).tolist()

    with np.load(resolve(args.stats), allow_pickle=False) as data:
        genes = [str(gene) for gene in data["genes"].tolist()]
    if args.gene not in genes:
        raise ValueError(f"{args.gene} is not present in adapter normalization genes")
    gene_index = genes.index(args.gene)

    rows: list[pd.DataFrame] = []
    for sample_id in final_test:
        sample_dir = resolve(args.tables_dir) / sample_id
        prediction_path = next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet"))
        target_path = next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet"))
        raw = pd.read_parquet(prediction_path)
        target = pd.read_parquet(target_path)
        coordinates = ["spot", "x", "y"]
        if not raw[coordinates].equals(target[coordinates]):
            raise ValueError(f"Prediction/target coordinates do not match for {sample_id}")

        adapter = np.load(resolve(args.adapter_dir) / f"{sample_id}.npy")
        if adapter.shape[0] != len(target) or adapter.shape[1] != len(genes):
            raise ValueError(f"Adapter prediction shape mismatch for {sample_id}: {adapter.shape}")

        frame = pd.DataFrame(
            {
                "sample_id": sample_id,
                "truth": target[args.gene].to_numpy(np.float64),
                "raw_prediction": raw[args.gene].to_numpy(np.float64),
                "adapter_prediction": adapter[:, gene_index].astype(np.float64),
            }
        )
        rows.append(frame)

    values = pd.concat(rows, ignore_index=True)
    truth = values["truth"].to_numpy()
    raw_prediction = values["raw_prediction"].to_numpy()
    adapter_prediction = values["adapter_prediction"].to_numpy()
    pooled_raw = metrics(truth, raw_prediction)
    pooled_adapter = metrics(truth, adapter_prediction)

    metric_rows = []
    for sample_id, frame in values.groupby("sample_id", sort=False):
        y = frame["truth"].to_numpy()
        raw_y = frame["raw_prediction"].to_numpy()
        adapter_y = frame["adapter_prediction"].to_numpy()
        raw_metrics = metrics(y, raw_y)
        adapter_metrics = metrics(y, adapter_y)
        metric_rows.extend(
            [
                {"sample_id": sample_id, "model": "raw_deepspotm", **raw_metrics},
                {"sample_id": sample_id, "model": "adapter", **adapter_metrics},
            ]
        )
    metric_rows.extend(
        [
            {"sample_id": "POOLED_FINAL_TEST", "model": "raw_deepspotm", **pooled_raw},
            {"sample_id": "POOLED_FINAL_TEST", "model": "adapter", **pooled_adapter},
        ]
    )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / f"{args.gene.lower()}_adapter_evaluation_metrics.csv", index=False)

    # Use shared limits so the raw and adapter parity panels are directly comparable.
    combined = np.concatenate([truth, raw_prediction, adapter_prediction])
    lo = float(np.nanpercentile(combined, 0.5))
    hi = float(np.nanpercentile(combined, 99.5))
    lo = min(0.0, lo)
    hi = max(hi, 1.0)
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    panels = [
        (axes[0], raw_prediction, "Raw DeepSpotM", "#6b7280", pooled_raw),
        (axes[1], adapter_prediction, "Adapter", "#b2182b", pooled_adapter),
    ]
    for ax, prediction, title, color, panel_metrics in panels:
        ax.hexbin(truth, prediction, gridsize=65, mincnt=1, bins="log", cmap="Blues", linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="#b2182b", linestyle="--", linewidth=1.2, label="identity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Xenium EGFR log1p(CP10K)")
        ax.set_ylabel("Predicted EGFR log1p(CP10K)")
        ax.set_title(title)
        add_stats(ax, "Held-out tiles", panel_metrics, (0.04, 0.96))

    raw_abs_error = np.abs(raw_prediction - truth)
    adapter_abs_error = np.abs(adapter_prediction - truth)
    error_ax = axes[2]
    error_ax.hexbin(
        np.log1p(raw_abs_error), np.log1p(adapter_abs_error),
        gridsize=65, mincnt=1, bins="log", cmap="Purples", linewidths=0,
    )
    error_lo = 0.0
    error_hi = float(np.nanpercentile(np.concatenate([raw_abs_error, adapter_abs_error]), 99.5))
    error_hi = max(error_hi, 1.0)
    error_ax.plot([0, np.log1p(error_hi)], [0, np.log1p(error_hi)], color="#b2182b", linestyle="--", linewidth=1.2)
    error_ax.set_xlim(0, np.log1p(error_hi))
    error_ax.set_ylim(0, np.log1p(error_hi))
    error_ax.set_xlabel("Raw DeepSpotM absolute error, log1p")
    error_ax.set_ylabel("Adapter absolute error, log1p")
    error_ax.set_title("Per-tile error: adapter vs raw")
    error_ax.text(
        0.04, 0.96,
        "Below diagonal = adapter is better\n"
        f"MAE: {pooled_raw['mae']:.3f} → {pooled_adapter['mae']:.3f}\n"
        f"RMSE: {pooled_raw['rmse']:.3f} → {pooled_adapter['rmse']:.3f}",
        transform=error_ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    fig.suptitle(
        f"Held-out {args.gene} adapter evaluation — {len(values):,} final-test tiles",
        fontsize=14, fontweight="bold",
    )
    figure_path = output_dir / f"{args.gene.lower()}_adapter_parity_and_improvement.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"saved={figure_path}")
    print(f"saved={output_dir / f'{args.gene.lower()}_adapter_evaluation_metrics.csv'}")
    print(pd.DataFrame(metric_rows).tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
