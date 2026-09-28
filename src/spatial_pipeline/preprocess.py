"""Per-sample schema validation, coordinate alignment, and matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .manifest import SampleRecord


REQUIRED_DEEPSPOT_COLUMNS = {"spot", "x", "y"}
REQUIRED_TRANSCRIPT_COLUMNS = {"feature_name", "x_location", "y_location", "qv"}
DEFAULT_CONTROL_PREFIXES = ("NegControl", "Unassigned", "BLANK", "NegPrb")


@dataclass
class ProcessedSample:
    """Aligned, unstandardized matrices for one sample."""

    sample_id: str
    split: str
    genes: list[str]
    predictions: np.ndarray
    targets_log1p_cp10k: np.ndarray
    x: np.ndarray
    y: np.ndarray


def validate_required_columns(frame: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def _decode_feature_names(values: pd.Series) -> pd.Series:
    def decode(value: object) -> str:
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)

    return values.map(decode)


def map_transcripts_to_tiles(
    sample: SampleRecord,
    deepspot: pd.DataFrame,
    transcripts: pd.DataFrame,
    min_qv: int = 20,
    control_prefixes: Sequence[str] = DEFAULT_CONTROL_PREFIXES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align transcripts to DeepSpot spots and return truth counts plus tile metadata."""

    validate_required_columns(deepspot, REQUIRED_DEEPSPOT_COLUMNS, "DeepSpot table")
    validate_required_columns(transcripts, REQUIRED_TRANSCRIPT_COLUMNS, "Xenium transcript table")
    if min_qv < 0:
        raise ValueError("min_qv must be non-negative")

    tx = transcripts.copy()
    tx["feature_name"] = _decode_feature_names(tx["feature_name"])
    tx = tx.loc[tx["qv"] >= min_qv].copy()
    if "is_gene" in tx.columns:
        tx = tx.loc[tx["is_gene"].astype(bool)].copy()
    else:
        control_mask = tx["feature_name"].str.startswith(tuple(control_prefixes))
        tx = tx.loc[~control_mask].copy()

    xp = tx["x_location"].to_numpy(dtype=np.float64) / sample.xenium_um_per_pixel
    yp = tx["y_location"].to_numpy(dtype=np.float64) / sample.xenium_um_per_pixel
    points = np.column_stack([xp, yp, np.ones(len(tx), dtype=np.float64)])
    try:
        inverse = np.linalg.inv(sample.affine_he_small_to_xenium)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"sample {sample.sample_id}: affine transform is not invertible") from exc
    small = (inverse @ points.T).T
    small_x = small[:, 0] / small[:, 2]
    small_y = small[:, 1] / small[:, 2]

    tx["he_x_full"] = small_x * sample.he_full_width_px / sample.he_small_width_px
    tx["he_y_full"] = small_y * sample.he_full_height_px / sample.he_small_height_px
    inside = (
        (tx["he_x_full"] >= 0)
        & (tx["he_x_full"] < sample.he_full_width_px)
        & (tx["he_y_full"] >= 0)
        & (tx["he_y_full"] < sample.he_full_height_px)
    )
    tx = tx.loc[inside].copy()
    tx["tile_x"] = (np.floor(tx["he_x_full"] / sample.tile_size_px) * sample.tile_size_px).astype(np.int64)
    tx["tile_y"] = (np.floor(tx["he_y_full"] / sample.tile_size_px) * sample.tile_size_px).astype(np.int64)

    tiles = deepspot[["spot", "x", "y"]].drop_duplicates().copy()
    lookup = {
        (int(round(row.x)), int(round(row.y))): row.spot
        for row in tiles.itertuples()
    }
    tx["spot"] = [
        lookup.get((int(tile_x), int(tile_y)))
        for tile_x, tile_y in zip(tx["tile_x"], tx["tile_y"])
    ]
    tx = tx.loc[tx["spot"].notna()].copy()

    counts = (
        tx.groupby(["spot", "feature_name"], observed=True)
        .size()
        .reset_index(name="count")
    )
    truth = counts.pivot(index="spot", columns="feature_name", values="count").fillna(0)
    return truth, tiles


def process_sample(
    sample: SampleRecord,
    deepspot: pd.DataFrame,
    transcripts: pd.DataFrame,
    min_qv: int = 20,
    cp10k: float = 10000.0,
) -> ProcessedSample:
    """Build one sample's raw DeepSpot and log1p-CP10K Xenium matrices."""

    if cp10k <= 0:
        raise ValueError("cp10k must be positive")
    truth, _ = map_transcripts_to_tiles(sample, deepspot, transcripts, min_qv=min_qv)
    metadata = {"spot", "x", "y", "patient"}
    deep_genes = [column for column in deepspot.columns if column not in metadata]
    genes = sorted(set(deep_genes).intersection(truth.columns))
    if not genes:
        raise ValueError(f"sample {sample.sample_id}: no shared genes")

    indexed = deepspot.set_index("spot")
    common_spots = [spot for spot in indexed.index if spot in truth.index]
    if not common_spots:
        raise ValueError(f"sample {sample.sample_id}: no matched DeepSpot/Xenium spots")
    predictions = indexed.loc[common_spots, genes].to_numpy(dtype=np.float32)
    counts = truth.reindex(common_spots).fillna(0).loc[:, genes].to_numpy(dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    targets = np.log1p(cp10k * counts / np.clip(totals, 1e-8, None)).astype(np.float32)
    coordinates = indexed.loc[common_spots, ["x", "y"]].to_numpy(dtype=np.float64)
    return ProcessedSample(
        sample_id=sample.sample_id,
        split=sample.split,
        genes=genes,
        predictions=predictions,
        targets_log1p_cp10k=targets,
        x=coordinates[:, 0],
        y=coordinates[:, 1],
    )


def load_and_process_sample(sample: SampleRecord, min_qv: int = 20) -> ProcessedSample:
    """Load exactly the two manifest-listed tables and process one sample."""

    deepspot = pd.read_parquet(sample.deepspot_path)
    transcripts = pd.read_parquet(sample.transcripts_path)
    return process_sample(sample, deepspot, transcripts, min_qv=min_qv)


def common_development_genes(samples: Sequence[ProcessedSample]) -> list[str]:
    """Derive the model vocabulary from development samples only."""

    development = [sample for sample in samples if sample.split == "development"]
    if not development:
        raise ValueError("at least one development sample is required")
    genes = set(development[0].genes)
    for sample in development[1:]:
        genes.intersection_update(sample.genes)
    result = sorted(genes)
    if not result:
        raise ValueError("development samples have no common genes")
    return result


def restrict_to_genes(sample: ProcessedSample, genes: Sequence[str]) -> ProcessedSample:
    """Reorder a sample to a development-defined vocabulary without imputation."""

    positions = {gene: index for index, gene in enumerate(sample.genes)}
    missing = [gene for gene in genes if gene not in positions]
    if missing:
        raise ValueError(f"sample {sample.sample_id} is missing required genes: {missing[:10]}")
    order = [positions[gene] for gene in genes]
    return ProcessedSample(
        sample_id=sample.sample_id,
        split=sample.split,
        genes=list(genes),
        predictions=sample.predictions[:, order],
        targets_log1p_cp10k=sample.targets_log1p_cp10k[:, order],
        x=sample.x.copy(),
        y=sample.y.copy(),
    )
