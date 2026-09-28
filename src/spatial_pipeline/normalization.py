"""Training-sample-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .preprocess import ProcessedSample


@dataclass
class NormalizationStats:
    prediction_mean: np.ndarray
    prediction_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def fit_training_stats(samples: Sequence[ProcessedSample]) -> NormalizationStats:
    """Fit gene-wise statistics from development samples only.

    The function rejects final-test samples so a caller cannot accidentally include
    their target distribution in normalization.
    """

    if not samples:
        raise ValueError("no training samples supplied")
    if any(sample.split != "development" for sample in samples):
        raise ValueError("normalization statistics may be fit only on development samples")
    genes = samples[0].predictions.shape[1]
    if any(sample.predictions.shape[1] != genes for sample in samples):
        raise ValueError("all samples must have the same gene count")
    predictions = np.concatenate([sample.predictions for sample in samples], axis=0)
    targets = np.concatenate([sample.targets_log1p_cp10k for sample in samples], axis=0)
    prediction_std = predictions.std(axis=0).astype(np.float32)
    target_std = targets.std(axis=0).astype(np.float32)
    prediction_std[prediction_std < 1e-6] = 1.0
    target_std[target_std < 1e-6] = 1.0
    return NormalizationStats(
        prediction_mean=predictions.mean(axis=0).astype(np.float32),
        prediction_std=prediction_std,
        target_mean=targets.mean(axis=0).astype(np.float32),
        target_std=target_std,
    )


def standardize_sample(sample: ProcessedSample, stats: NormalizationStats) -> tuple[np.ndarray, np.ndarray]:
    """Apply persisted development statistics to any sample."""

    if sample.predictions.shape[1] != len(stats.prediction_mean):
        raise ValueError("sample gene count does not match normalization statistics")
    predictions = ((sample.predictions - stats.prediction_mean) / stats.prediction_std).astype(np.float32)
    targets = ((sample.targets_log1p_cp10k - stats.target_mean) / stats.target_std).astype(np.float32)
    return predictions, targets
