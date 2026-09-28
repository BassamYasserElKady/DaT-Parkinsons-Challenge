"""Per-sample and pooled evaluation metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def gene_wise_spatial_pearson(prediction: np.ndarray, target: np.ndarray, genes: Sequence[str]) -> pd.DataFrame:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != len(genes):
        raise ValueError("prediction, target, and genes have incompatible shapes")
    return pd.DataFrame(
        {
            "gene": list(genes),
            "spatial_pearson": [
                _safe_correlation(prediction[:, index], target[:, index], "pearson")
                for index in range(prediction.shape[1])
            ],
        }
    )


def per_tile_gene_rank_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    values = [
        _safe_correlation(prediction[row], target[row], "spearman")
        for row in range(prediction.shape[0])
    ]
    return float(np.nanmean(values)) if values else float("nan")


def aggregate_abundance_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    pred_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    return {
        "aggregate_pearson": _safe_correlation(pred_mean, target_mean, "pearson"),
        "aggregate_spearman": _safe_correlation(pred_mean, target_mean, "spearman"),
    }


def evaluate_arrays(
    sample_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    genes: Sequence[str],
) -> dict[str, float | str | int]:
    per_gene = gene_wise_spatial_pearson(prediction, target, genes)
    aggregate = aggregate_abundance_metrics(prediction, target)
    return {
        "sample_id": sample_id,
        "n_tiles": int(prediction.shape[0]),
        "n_genes": int(prediction.shape[1]),
        "mean_gene_spatial_pearson": float(per_gene["spatial_pearson"].mean()),
        "per_tile_gene_rank_spearman": per_tile_gene_rank_spearman(prediction, target),
        **aggregate,
    }


def evaluate_per_sample_and_pooled(
    samples: Sequence[tuple[str, np.ndarray, np.ndarray, Sequence[str]]],
) -> pd.DataFrame:
    """Return four independent sample rows plus one pooled row."""

    if not samples:
        raise ValueError("no samples supplied")
    rows = [evaluate_arrays(sample_id, prediction, target, genes) for sample_id, prediction, target, genes in samples]
    reference_genes = list(samples[0][3])
    pooled_prediction = []
    pooled_target = []
    for sample_id, prediction, target, genes in samples:
        if list(genes) != reference_genes:
            raise ValueError(f"gene order mismatch for pooled evaluation: {sample_id}")
        pooled_prediction.append(prediction)
        pooled_target.append(target)
    rows.append(
        evaluate_arrays(
            "POOLED_FINAL_TEST",
            np.concatenate(pooled_prediction, axis=0),
            np.concatenate(pooled_target, axis=0),
            reference_genes,
        )
    )
    return pd.DataFrame(rows)
