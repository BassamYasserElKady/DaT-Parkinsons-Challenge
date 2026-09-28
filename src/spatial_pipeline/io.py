"""Small, explicit persistence helpers for processed per-sample matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .normalization import NormalizationStats
from .preprocess import ProcessedSample


def save_processed_sample(sample: ProcessedSample, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sample.sample_id}.npz"
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample.sample_id),
        split=np.asarray(sample.split),
        genes=np.asarray(sample.genes),
        predictions=sample.predictions,
        targets_log1p_cp10k=sample.targets_log1p_cp10k,
        x=sample.x,
        y=sample.y,
    )
    return path


def load_processed_sample(path: Path) -> ProcessedSample:
    with np.load(path, allow_pickle=False) as data:
        return ProcessedSample(
            sample_id=str(data["sample_id"].item()),
            split=str(data["split"].item()),
            genes=[str(gene) for gene in data["genes"].tolist()],
            predictions=data["predictions"].astype(np.float32),
            targets_log1p_cp10k=data["targets_log1p_cp10k"].astype(np.float32),
            x=data["x"].astype(np.float64),
            y=data["y"].astype(np.float64),
        )


def save_normalization_stats(stats: NormalizationStats, genes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        genes=np.asarray(genes),
        prediction_mean=stats.prediction_mean,
        prediction_std=stats.prediction_std,
        target_mean=stats.target_mean,
        target_std=stats.target_std,
    )
