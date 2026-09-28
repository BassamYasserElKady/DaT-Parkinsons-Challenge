#!/usr/bin/env python3
"""Find Xenium tile-level gene patterns that may hint at EGFR presence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]


def correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan, np.nan
    return float(pearsonr(x[valid], y[valid]).statistic), float(spearmanr(x[valid], y[valid]).statistic)


def sample_gene_metrics(frame: pd.DataFrame, sample_id: str, split: str, genes: list[str]) -> pd.DataFrame:
    egfr = frame["EGFR"].to_numpy(np.float64)
    positive = egfr > 0
    rows = []
    for gene in genes:
        if gene == "EGFR":
            continue
        values = frame[gene].to_numpy(np.float64)
        pearson, spearman = correlations(values, egfr)
        mean_positive = float(values[positive].mean()) if positive.any() else np.nan
        mean_zero = float(values[~positive].mean()) if (~positive).any() else np.nan
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "gene": gene,
                "tiles": len(frame),
                "egfr_positive_fraction": float(positive.mean()),
                "gene_nonzero_fraction": float(np.mean(values > 0)),
                "gene_mean_in_egfr_positive_tiles": mean_positive,
                "gene_mean_in_egfr_zero_tiles": mean_zero,
                "positive_minus_zero_mean": mean_positive - mean_zero,
                "spatial_pearson_with_egfr": pearson,
                "spatial_spearman_with_egfr": spearman,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_tile_hints"))
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = pd.read_csv(args.manifest if args.manifest.is_absolute() else root / args.manifest)
    sample_rows = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        sample_dir = root / args.tables_dir / sample_id
        target_path = next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet"))
        target = pd.read_parquet(target_path)
        genes = [column for column in target.columns if column not in {"spot", "x", "y"}]
        sample_rows.append(sample_gene_metrics(target, sample_id, str(row.split), genes))
    by_sample = pd.concat(sample_rows, ignore_index=True)

    dev = by_sample[by_sample["split"] == "development"].copy()
    test = by_sample[by_sample["split"] == "final_test"].copy()
    summary = dev.groupby("gene", as_index=False).agg(
        development_samples=("sample_id", "nunique"),
        mean_pearson=("spatial_pearson_with_egfr", "mean"),
        median_pearson=("spatial_pearson_with_egfr", "median"),
        mean_spearman=("spatial_spearman_with_egfr", "mean"),
        median_spearman=("spatial_spearman_with_egfr", "median"),
        mean_positive_minus_zero=("positive_minus_zero_mean", "mean"),
        sign_consistency_pearson=("spatial_pearson_with_egfr", lambda x: float(np.mean(x > 0))),
        sign_consistency_spearman=("spatial_spearman_with_egfr", lambda x: float(np.mean(x > 0))),
    )
    summary["stable_positive_hint"] = (summary["sign_consistency_pearson"] >= 0.75) & (summary["sign_consistency_spearman"] >= 0.75) & (summary["mean_spearman"] > 0)
    summary["stable_negative_hint"] = (summary["sign_consistency_pearson"] <= 0.25) & (summary["sign_consistency_spearman"] <= 0.25) & (summary["mean_spearman"] < 0)
    summary["rank_score"] = summary["mean_spearman"].abs() * summary[["sign_consistency_pearson", "sign_consistency_spearman"]].min(axis=1)
    positive = summary[summary["stable_positive_hint"]].sort_values("rank_score", ascending=False).head(args.top_n)
    negative = summary[summary["stable_negative_hint"]].sort_values("rank_score", ascending=False).head(args.top_n)

    test_summary = test.groupby("gene", as_index=False).agg(
        final_test_mean_pearson=("spatial_pearson_with_egfr", "mean"),
        final_test_mean_spearman=("spatial_spearman_with_egfr", "mean"),
        final_test_positive_minus_zero=("positive_minus_zero_mean", "mean"),
    )
    positive = positive.merge(test_summary, on="gene", how="left")
    negative = negative.merge(test_summary, on="gene", how="left")

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    by_sample.to_csv(output / "gene_egfr_correlations_by_sample.csv", index=False)
    summary.to_csv(output / "development_hint_summary.csv", index=False)
    positive.to_csv(output / "stable_positive_hints.csv", index=False)
    negative.to_csv(output / "stable_negative_hints.csv", index=False)

    report = [
        "# Xenium tile-level hints for EGFR",
        "",
        "Candidate hints were discovered using only the eight development samples. A hint is a gene whose tile-level Xenium pattern has consistent positive or negative EGFR association in at least 75% of development samples.",
        "",
        "## Stable positive candidates",
        "",
        positive[["gene", "mean_pearson", "mean_spearman", "sign_consistency_pearson", "sign_consistency_spearman", "final_test_mean_pearson", "final_test_mean_spearman"]].to_markdown(index=False) if len(positive) else "None met the stability threshold.",
        "",
        "## Stable negative candidates",
        "",
        negative[["gene", "mean_pearson", "mean_spearman", "sign_consistency_pearson", "sign_consistency_spearman", "final_test_mean_pearson", "final_test_mean_spearman"]].to_markdown(index=False) if len(negative) else "None met the stability threshold.",
        "",
        "These are spatial co-occurrence hints, not causal markers or cell-type labels. They should be used as candidate inputs or auxiliary targets only after validation inside development folds.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    print(f"output={output}")
    print("stable positive hints:")
    print(positive[["gene", "mean_pearson", "mean_spearman", "final_test_mean_pearson", "final_test_mean_spearman"]].to_string(index=False))
    print("stable negative hints:")
    print(negative[["gene", "mean_pearson", "mean_spearman", "final_test_mean_pearson", "final_test_mean_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
