#!/usr/bin/env python3
"""Compare pre-registration nonempty axial slice counts with OOF errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RAW_COUNTS = ROOT / "results/raw_adaptive_hotspot_slices_017_leading4_parallel/adaptive_slice_selection.csv"
OOF = ROOT / "results/cv5_submission_flip_clip/oof_predictions.csv"
OUTPUT = ROOT / "results/raw_nonempty_slices_vs_oof_errors"


def group_summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
    }


def main() -> None:
    raw = pd.read_csv(RAW_COUNTS)
    oof = pd.read_csv(OOF)
    if raw["uid"].duplicated().any() or oof["uid"].duplicated().any():
        raise ValueError("UIDs must be unique in both inputs")

    columns = [
        "uid",
        "n_slices",
        "nonempty_slices",
        "discarded_leading_nonempty_slices",
        "eligible_nonempty_slices",
    ]
    merged = oof.merge(raw[columns], on="uid", how="left", validate="one_to_one")
    if merged["nonempty_slices"].isna().any():
        raise ValueError("Some OOF scans have no raw slice-count record")

    merged["misclassified"] = merged["actual"].ne(merged["prediction"])
    probability = merged["probability"].clip(1e-15, 1 - 1e-15)
    merged["scan_log_loss"] = -(
        merged["actual"] * np.log(probability)
        + (1 - merged["actual"]) * np.log(1 - probability)
    )
    merged["classification_group"] = np.where(merged["misclassified"], "misclassified", "correct")

    wrong = merged.loc[merged["misclassified"], "nonempty_slices"]
    correct = merged.loc[~merged["misclassified"], "nonempty_slices"]
    mann_whitney = stats.mannwhitneyu(wrong, correct, alternative="two-sided")
    welch = stats.ttest_ind(wrong, correct, equal_var=False)
    point_biserial = stats.pointbiserialr(merged["misclassified"].astype(int), merged["nonempty_slices"])
    spearman = stats.spearmanr(merged["nonempty_slices"], merged["scan_log_loss"])

    n1, n0 = len(wrong), len(correct)
    pooled_sd = np.sqrt(((n1 - 1) * wrong.var(ddof=1) + (n0 - 1) * correct.var(ddof=1)) / (n1 + n0 - 2))
    cohens_d = float((wrong.mean() - correct.mean()) / pooled_sd)
    error_auc = float(roc_auc_score(merged["misclassified"].astype(int), merged["nonempty_slices"]))

    distribution = (
        merged.groupby("nonempty_slices", as_index=False)
        .agg(total_scans=("uid", "size"), misclassified_scans=("misclassified", "sum"))
        .sort_values("nonempty_slices")
    )
    distribution["correct_scans"] = distribution["total_scans"] - distribution["misclassified_scans"]
    distribution["misclassification_rate"] = distribution["misclassified_scans"] / distribution["total_scans"]
    distribution["share_of_all_misclassified"] = distribution["misclassified_scans"] / int(merged["misclassified"].sum())
    distribution["share_of_all_correct"] = distribution["correct_scans"] / int((~merged["misclassified"]).sum())

    # Fixed-width bins make the comparison readable despite many exact count values.
    edges = list(range(30, 141, 10))
    labels = [f"{left}-{right - 1}" for left, right in zip(edges[:-1], edges[1:])]
    merged["nonempty_slice_bin"] = pd.cut(
        merged["nonempty_slices"], bins=edges, labels=labels, right=False, include_lowest=True
    )
    binned = (
        merged.groupby("nonempty_slice_bin", observed=False, as_index=False)
        .agg(total_scans=("uid", "size"), misclassified_scans=("misclassified", "sum"))
    )
    binned["correct_scans"] = binned["total_scans"] - binned["misclassified_scans"]
    binned["misclassification_rate"] = np.where(
        binned["total_scans"] > 0,
        binned["misclassified_scans"] / binned["total_scans"],
        np.nan,
    )

    summary = {
        "definition": "An axial slice in the raw, unregistered NIfTI is nonempty when its maximum voxel intensity is greater than zero.",
        "all_scans": group_summary(merged["nonempty_slices"]),
        "misclassified": group_summary(wrong),
        "correct": group_summary(correct),
        "difference_wrong_minus_correct_mean": float(wrong.mean() - correct.mean()),
        "cohens_d": cohens_d,
        "mann_whitney_u": float(mann_whitney.statistic),
        "mann_whitney_p": float(mann_whitney.pvalue),
        "welch_t": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "point_biserial_r_error_vs_count": float(point_biserial.statistic),
        "point_biserial_p": float(point_biserial.pvalue),
        "auc_count_predicting_misclassification": error_auc,
        "auc_direction_free": max(error_auc, 1 - error_auc),
        "spearman_count_vs_scan_log_loss_rho": float(spearman.statistic),
        "spearman_count_vs_scan_log_loss_p": float(spearman.pvalue),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT / "per_scan_raw_nonempty_slices_and_oof.csv", index=False)
    distribution.to_csv(OUTPUT / "exact_count_distribution.csv", index=False)
    binned.to_csv(OUTPUT / "binned_count_distribution.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nBINNED DISTRIBUTION")
    print(binned.to_string(index=False))
    print("\nMOST COMMON EXACT COUNTS")
    print(distribution.sort_values("total_scans", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
