#!/usr/bin/env python3
"""Quick development-only LOSO test of four-gene context for EGFR.

The four genes are fixed from the prior development-derived hint analysis:
LAMB1, LAMA2, PDGFRB, and C1R.  For each development sample, a ridge model
is fit on the other development samples and evaluated on the held-out sample.
Final-test samples are never loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]

HINT_GENES = ["LAMB1", "LAMA2", "PDGFRB", "C1R"]


def corr(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    return (
        float(pearsonr(pred, target).statistic),
        float(spearmanr(pred, target).statistic),
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardized ridge with an unpenalized intercept."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    penalty = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    penalty[1:, 1:] = np.eye(design.shape[1] - 1) * alpha
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef, np.column_stack([mean, scale])


def ridge_predict(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    coef, stats = fit
    mean, scale = stats[:, 0], stats[:, 1]
    z = (x - mean) / scale
    return coef[0] + z @ coef[1:]


def load_sample(root: Path, tables_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    sample_dir = root / tables_dir / sample_id
    prediction_path = next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet"))
    target_path = next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet"))
    prediction = pd.read_parquet(prediction_path)
    target = pd.read_parquet(target_path)
    if not prediction[["spot", "x", "y"]].equals(target[["spot", "x", "y"]]):
        raise ValueError(f"coordinate mismatch in {sample_id}")
    required = ["EGFR", *HINT_GENES]
    missing = [gene for gene in required if gene not in prediction or gene not in target]
    if missing:
        raise ValueError(f"missing genes in {sample_id}: {missing}")
    features = prediction[required].to_numpy(np.float64)
    y = target["EGFR"].to_numpy(np.float64)
    return features, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_hint_loso"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = pd.read_csv(manifest_path)
    development = manifest.loc[manifest["split"].eq("development"), "sample_id"].astype(str).tolist()
    if not development:
        raise ValueError("manifest contains no development samples")

    samples = {sample_id: load_sample(root, args.tables_dir, sample_id) for sample_id in development}
    rows: list[dict[str, object]] = []
    for held_out in development:
        train_ids = [sample_id for sample_id in development if sample_id != held_out]
        x_train = np.concatenate([samples[sample_id][0] for sample_id in train_ids])
        y_train = np.concatenate([samples[sample_id][1] for sample_id in train_ids])
        x_test, y_test = samples[held_out]

        raw_pearson, raw_spearman = corr(x_test[:, 0], y_test)
        # The model is deliberately small: raw EGFR plus the four context genes.
        fit = ridge_fit(x_train, y_train, alpha=args.alpha)
        module_pred = ridge_predict(x_test, fit)
        module_pearson, module_spearman = corr(module_pred, y_test)

        # A one-dimensional module score gives an easy-to-read ablation.
        module_score_train = x_train[:, 1:].mean(axis=1, keepdims=True)
        module_score_test = x_test[:, 1:].mean(axis=1, keepdims=True)
        score_fit = ridge_fit(np.column_stack([x_train[:, 0], module_score_train]), y_train, alpha=args.alpha)
        score_pred = ridge_predict(np.column_stack([x_test[:, 0], module_score_test]), score_fit)
        score_pearson, score_spearman = corr(score_pred, y_test)

        rows.extend(
            [
                {"held_out_sample": held_out, "model": "raw_deepspot_egfr", "tiles": len(y_test), "pearson": raw_pearson, "spearman": raw_spearman},
                {"held_out_sample": held_out, "model": "ridge_raw_egfr_plus_4_hints", "tiles": len(y_test), "pearson": module_pearson, "spearman": module_spearman},
                {"held_out_sample": held_out, "model": "ridge_raw_egfr_plus_mean_hint_score", "tiles": len(y_test), "pearson": score_pearson, "spearman": score_spearman},
            ]
        )

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model", as_index=False)[["pearson", "spearman"]].mean()
    raw = summary.loc[summary["model"].eq("raw_deepspot_egfr")].iloc[0]
    summary["pearson_gain_vs_raw"] = summary["pearson"] - float(raw["pearson"])
    summary["spearman_gain_vs_raw"] = summary["spearman"] - float(raw["spearman"])

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary.to_csv(output_dir / "mean_metrics.csv", index=False)
    report = [
        "# EGFR hint-gene LOSO test",
        "",
        "This exploratory test uses development samples only. Each sample is held out once; the four final-test samples are not loaded.",
        "",
        f"Fixed hint genes: {', '.join(HINT_GENES)}",
        f"Ridge alpha: {args.alpha}",
        "",
        "## Mean held-out metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "Interpretation: a positive gain means the fixed hint genes improved EGFR spatial agreement on unseen development samples. This is not a final test result and does not establish causal biology.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"outputs={output_dir}")


if __name__ == "__main__":
    main()
