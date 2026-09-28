#!/usr/bin/env python3
"""Build a cross-sample Xenium cell-composition atlas for EGFR.

The atlas is deliberately separated from the EGFR target used in the tile
models.  Cell states are clustered from marker-score features that do not use
EGFR; EGFR is added only afterward for characterization and prediction tests.

Outputs include cell-state composition by sample and tile, EGFR summaries by
state, marker-score descriptions of each state, and leakage-aware tests of
whether tile composition predicts Xenium EGFR.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.spatial_pipeline.xenium_archive import NestedZipStore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "sample_manifest.csv"
ANNOTATION_ROOT = ROOT / "results" / "xenium_egfr_cell_annotation"
TABLE_ROOT = ROOT / "results" / "deepspot_tables"
OUT_ROOT = ROOT / "results" / "xenium_egfr_cell_atlas"
TILE_SIZE = 224

# These scores are derived from panel markers and deliberately exclude the
# basal/stem scores because those marker sets contain EGFR itself.
STATE_SCORES = [
    "score_tumor_epithelial",
    "score_myoepithelial",
    "score_caf_stromal",
    "score_endothelial",
    "score_perivascular",
    "score_myeloid",
    "score_t_cell",
    "score_b_cell",
    "score_plasma",
    "score_mast",
    "score_adipocyte",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--sample-per-state-fit", type=int, default=6000)
    p.add_argument("--random-state", type=int, default=17)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    return p.parse_args()


def manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST_PATH)
    if "sample" not in frame and "sample_id" in frame:
        frame = frame.rename(columns={"sample_id": "sample"})
    return frame


def read_centroids(path: Path) -> np.ndarray:
    """Read only cell centroids; do not load the expression matrix again."""
    with zipfile.ZipFile(path) as outer:
        with NestedZipStore(outer, "cells.zarr.zip") as root:
            summary = root["cell_summary"]
            names = [v.decode() if isinstance(v, bytes) else str(v) for v in summary.attrs["column_names"]]
            x_i, y_i = names.index("cell_centroid_x"), names.index("cell_centroid_y")
            return np.asarray(summary[:, [x_i, y_i]], dtype=np.float64)


def affine(value: str) -> np.ndarray:
    return np.asarray([float(v.strip()) for v in str(value).split(";")], dtype=float).reshape(3, 3)


def cell_tiles(row: pd.Series, centroids_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xenium_px = centroids_um / float(row["xenium_pixel_size_um"])
    points = np.column_stack([xenium_px, np.ones(len(xenium_px))])
    mapped = (np.linalg.inv(affine(row["alignment_affine_row_major"])) @ points.T).T
    x, y = mapped[:, 0] / mapped[:, 2], mapped[:, 1] / mapped[:, 2]
    inside = (
        (x >= 0) & (x < float(row["he_width_px"]))
        & (y >= 0) & (y < float(row["he_height_px"]))
    )
    tile_x = np.floor(x / TILE_SIZE).astype(np.int64) * TILE_SIZE
    tile_y = np.floor(y / TILE_SIZE).astype(np.int64) * TILE_SIZE
    complete = inside & (tile_x + TILE_SIZE <= int(row["he_width_px"])) & (tile_y + TILE_SIZE <= int(row["he_height_px"]))
    return tile_x, tile_y, complete


def sample_annotation(sample: str) -> pd.DataFrame:
    path = ANNOTATION_ROOT / f"{sample}_cell_scores.parquet"
    cols = ["sample", "cell_index", "gene_transcripts", "egfr_count", "egfr_log1p_cp10k", "predicted_lineage", *STATE_SCORES]
    frame = pd.read_parquet(path, columns=cols)
    for col in STATE_SCORES:
        frame[col] = frame[col].astype(np.float32)
    return frame


def fit_states(annotations: dict[str, pd.DataFrame], n_clusters: int, per_sample: int, random_state: int):
    pieces = []
    rng = np.random.default_rng(random_state)
    for sample, frame in annotations.items():
        n = min(len(frame), per_sample)
        idx = rng.choice(len(frame), size=n, replace=False)
        pieces.append(frame.iloc[idx][STATE_SCORES].to_numpy(np.float32))
    fit = np.concatenate(pieces, axis=0)
    scaler = StandardScaler().fit(fit)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init=10,
        max_iter=100,
    ).fit(scaler.transform(fit))
    return scaler, model


def state_names(scaler: StandardScaler, model: MiniBatchKMeans) -> dict[int, str]:
    centers = scaler.inverse_transform(model.cluster_centers_)
    dominant = np.argmax(centers, axis=1)
    counts: dict[str, int] = {}
    names = {}
    for i, j in enumerate(dominant):
        base = STATE_SCORES[j].removeprefix("score_")
        counts[base] = counts.get(base, 0) + 1
        suffix = f"_{counts[base]}" if counts[base] > 1 else ""
        names[i] = f"{base}{suffix}"
    return names


def assign_cells(frame: pd.DataFrame, scaler: StandardScaler, model: MiniBatchKMeans, names: dict[int, str]) -> pd.DataFrame:
    out = frame.copy()
    labels = model.predict(scaler.transform(out[STATE_SCORES].to_numpy(np.float32)))
    out["state_id"] = labels.astype(np.int16)
    out["state"] = pd.Series(labels, index=out.index).map(names).astype(str)
    return out


def summarize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, state_id, state), g in cells.groupby(["sample", "state_id", "state"], sort=True):
        rows.append({
            "sample": sample, "state_id": int(state_id), "state": state,
            "cells": int(len(g)), "fraction_of_sample": float(len(g) / len(cells[cells["sample"].eq(sample)])),
            "egfr_positive_cells": int((g.egfr_count > 0).sum()),
            "egfr_detection_fraction": float((g.egfr_count > 0).mean()),
            "egfr_mean_log1p_cp10k": float(g.egfr_log1p_cp10k.mean()),
            "egfr_median_log1p_cp10k": float(g.egfr_log1p_cp10k.median()),
            "egfr_mean_count": float(g.egfr_count.mean()),
            "median_gene_transcripts": float(g.gene_transcripts.median()),
        })
    return pd.DataFrame(rows)


def state_profiles(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, state), g in cells.groupby(["state_id", "state"], sort=True):
        row = {"state_id": int(state_id), "state": state, "cells": int(len(g))}
        for col in STATE_SCORES:
            row[f"mean_{col}"] = float(g[col].mean())
        for label, frac in g["predicted_lineage"].value_counts(normalize=True).items():
            row[f"lineage_fraction_{label}"] = float(frac)
        rows.append(row)
    return pd.DataFrame(rows)


def tile_atlas(row: pd.Series, cells: pd.DataFrame, centroids: np.ndarray, state_count: int) -> pd.DataFrame:
    tx, ty, keep = cell_tiles(row, centroids)
    cells = cells.iloc[np.flatnonzero(keep)].copy()
    cells["tile_x"] = tx[keep]
    cells["tile_y"] = ty[keep]
    keys = ["tile_x", "tile_y"]
    totals = cells.groupby(keys, as_index=False).agg(n_cells=("cell_index", "size"), xenium_egfr_counts=("egfr_count", "sum"), xenium_egfr_log1p_cp10k=("egfr_log1p_cp10k", "mean"))
    state_counts = pd.crosstab([cells.tile_x, cells.tile_y], cells.state_id).reindex(columns=range(state_count), fill_value=0)
    state_counts = state_counts.div(state_counts.sum(axis=1), axis=0).fillna(0).reset_index()
    state_counts.columns = ["tile_x", "tile_y", *[f"state_fraction_{i}" for i in range(state_count)]]
    lineage_counts = pd.crosstab([cells.tile_x, cells.tile_y], cells.predicted_lineage)
    lineage_counts = lineage_counts.div(lineage_counts.sum(axis=1), axis=0).fillna(0).reset_index()
    lineage_counts.columns = ["tile_x", "tile_y", *[f"lineage_fraction_{str(c)}" for c in lineage_counts.columns[2:]]]
    out = totals.merge(state_counts, on=keys, how="left").merge(lineage_counts, on=keys, how="left")
    out.insert(0, "sample", row["sample"])
    return out


def load_tile_target(sample: str) -> pd.DataFrame:
    directory = TABLE_ROOT / sample
    target = pd.read_parquet(directory / f"{sample.lower()}_xenium_log1p_cp10k_targets.parquet", columns=["spot", "x", "y", "EGFR"])
    target = target.rename(columns={"x": "tile_x", "y": "tile_y", "EGFR": "target_egfr_log1p_cp10k"})
    target.insert(0, "sample", sample)
    return target.drop(columns="spot")


def metric(y: np.ndarray, pred: np.ndarray, method: str, group: str) -> dict:
    row = {"group": group, "method": method, "n_tiles": int(len(y)), "rmse": float(np.sqrt(np.mean((pred-y)**2))), "mae": float(np.mean(np.abs(pred-y)))}
    row["pearson_r"] = float(pearsonr(y, pred).statistic) if np.unique(y).size > 1 and np.unique(pred).size > 1 else np.nan
    row["spearman_r"] = float(spearmanr(y, pred).statistic) if np.unique(y).size > 1 and np.unique(pred).size > 1 else np.nan
    return row


def composition_model(atlas: pd.DataFrame, manifest_frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    feature_cols = [c for c in atlas.columns if c.startswith("state_fraction_")]
    dev = manifest_frame.loc[manifest_frame.split.eq("development"), "sample"].tolist()
    test = manifest_frame.loc[manifest_frame.split.eq("final_test"), "sample"].tolist()
    rows = []
    for held in dev:
        train = atlas[atlas["sample"].isin([s for s in dev if s != held])]
        valid = atlas[atlas["sample"].eq(held)]
        scaler = StandardScaler().fit(train[feature_cols])
        model = Ridge(alpha=alpha).fit(scaler.transform(train[feature_cols]), train.target_egfr_log1p_cp10k)
        pred = np.maximum(model.predict(scaler.transform(valid[feature_cols])), 0)
        rows.append(metric(valid.target_egfr_log1p_cp10k.to_numpy(), pred, "composition_only_ridge", held))
    train = atlas[atlas["sample"].isin(dev)]
    final = atlas[atlas["sample"].isin(test)]
    scaler = StandardScaler().fit(train[feature_cols])
    model = Ridge(alpha=alpha).fit(scaler.transform(train[feature_cols]), train.target_egfr_log1p_cp10k)
    pred = np.maximum(model.predict(scaler.transform(final[feature_cols])), 0)
    rows.append(metric(final.target_egfr_log1p_cp10k.to_numpy(), pred, "composition_only_ridge", "final_test_pooled"))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mf = manifest()
    sample_rows = {r.sample: r for r in mf.itertuples(index=False)}
    samples = mf["sample"].astype(str).tolist()
    annotations = {sample: sample_annotation(sample) for sample in samples}
    scaler, model = fit_states(annotations, args.n_clusters, args.sample_per_state_fit, args.random_state)
    names = state_names(scaler, model)
    assigned = {sample: assign_cells(frame, scaler, model, names) for sample, frame in annotations.items()}
    all_cells = pd.concat(assigned.values(), ignore_index=True)
    summarize_cells(all_cells).to_csv(out / "cell_state_egfr_summary.csv", index=False)
    state_profiles(all_cells).to_csv(out / "cell_state_profiles.csv", index=False)

    tile_parts = []
    for sample in samples:
        print(f"Mapping {sample} cells to tiles ...", flush=True)
        row = mf[mf["sample"].eq(sample)].iloc[0]
        archive = ROOT / str(row.xenium_archive_path)
        coords = read_centroids(archive)
        tile_parts.append(tile_atlas(row, assigned[sample], coords, args.n_clusters))
    tile_atlas_frame = pd.concat(tile_parts, ignore_index=True)
    targets = pd.concat([load_tile_target(sample) for sample in samples], ignore_index=True)
    tile_atlas_frame = tile_atlas_frame.merge(targets, on=["sample", "tile_x", "tile_y"], how="inner")
    tile_atlas_frame.to_csv(out / "tile_composition_egfr_atlas.csv", index=False)
    composition_model(tile_atlas_frame, mf, args.alpha).to_csv(out / "composition_model_metrics.csv", index=False)

    centers = scaler.inverse_transform(model.cluster_centers_)
    pd.DataFrame(centers, columns=STATE_SCORES).assign(state_id=range(args.n_clusters), state=lambda d: d.state_id.map(names)).to_csv(out / "state_centers.csv", index=False)
    metadata = {
        "n_samples": len(samples), "samples": samples, "n_clusters": args.n_clusters,
        "cluster_features": STATE_SCORES,
        "egfr_used_for_clustering": False,
        "tile_size_px": TILE_SIZE,
        "cell_labels": "marker-score state clusters; predicted_lineage is a transparent panel-aware annotation",
        "target": "Xenium tile EGFR log1p(CP10K)",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"out": str(out), "cells": len(all_cells), "tiles": len(tile_atlas_frame), "states": names}, indent=2))


if __name__ == "__main__":
    main()
