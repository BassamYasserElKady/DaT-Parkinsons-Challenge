#!/usr/bin/env python3
"""Cell-level niche analysis for the S2-Bot HER2 3+ Xenium sample.

The raw Xenium archive is read in place. Nested Zarr members are decoded from
memory; no raw files are extracted or modified.

Positive-cell rules are deliberately explicit and conservative:
  protective macrophage: CD68 > 0 AND CD163 > 0 AND CXCL10 > 0
  myCAF: ACTA2 > 0 AND TAGLN > 0
  vascular element: CXCL12 > 0 OR RGS5 > 0
  EGFR-positive tumor target: EGFR > 0

Counts are normalized to log1p(CP10K) for ligand-gradient summaries.
Coordinates are Xenium cell centroids in microns. Ripley's cross-K is an
uncorrected rectangular-window estimate, with a label-permutation null.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numcodecs import get_codec
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.cluster import DBSCAN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT
    / "data/raw/10x/section_2_bot/"
    / "Human_Breast_Biomarkers_S2_Bot_xe_outs.zip"
)
DEFAULT_OUTPUT = ROOT / "results/s2_bot_niche_analysis"

MARKERS = (
    "CD68",
    "CD163",
    "CXCL10",
    "ACTA2",
    "TAGLN",
    "CXCL12",
    "RGS5",
    "EGFR",
    "PTN",
    "CCL5",
)


class NestedZarr:
    """Minimal read-only Zarr v2 reader for a nested ZIP member."""

    def __init__(self, outer: ZipFile, member: str):
        self.inner = ZipFile(BytesIO(outer.read(member)))

    def close(self) -> None:
        self.inner.close()

    def __enter__(self) -> "NestedZarr":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def meta(self, array_name: str) -> dict:
        return json.loads(self.inner.read(array_name + "/.zarray"))

    def attrs(self, group_name: str) -> dict:
        return json.loads(self.inner.read(group_name + "/.zattrs"))

    def chunk(self, array_name: str, chunk_index: tuple[int, ...], meta: dict) -> np.ndarray:
        key = array_name + "/" + ".".join(str(i) for i in chunk_index)
        raw = self.inner.read(key)
        if meta.get("compressor"):
            raw = get_codec(meta["compressor"]).decode(raw)
        shape = tuple(
            min(meta["chunks"][axis], meta["shape"][axis] - chunk_index[axis] * meta["chunks"][axis])
            for axis in range(len(meta["shape"]))
        )
        return np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(shape, order=meta.get("order", "C"))

    def read_array(self, array_name: str) -> np.ndarray:
        meta = self.meta(array_name)
        shape = tuple(meta["shape"])
        out = np.empty(shape, dtype=np.dtype(meta["dtype"]))
        grid = [range((shape[axis] + meta["chunks"][axis] - 1) // meta["chunks"][axis]) for axis in range(len(shape))]
        import itertools

        for idx in itertools.product(*grid):
            block = self.chunk(array_name, idx, meta)
            starts = tuple(idx[axis] * meta["chunks"][axis] for axis in range(len(shape)))
            slices = tuple(slice(starts[axis], starts[axis] + block.shape[axis]) for axis in range(len(shape)))
            out[slices] = block
        return out

    def read_1d_slice(self, array_name: str, start: int, stop: int) -> np.ndarray:
        meta = self.meta(array_name)
        chunk_len = meta["chunks"][0]
        out = np.empty(stop - start, dtype=np.dtype(meta["dtype"]))
        cursor = 0
        first = start // chunk_len
        last = (stop - 1) // chunk_len if stop > start else first
        for chunk_number in range(first, last + 1):
            block = self.chunk(array_name, (chunk_number,), meta)
            block_start = chunk_number * chunk_len
            lo = max(start, block_start) - block_start
            hi = min(stop, block_start + len(block)) - block_start
            out[cursor : cursor + (hi - lo)] = block[lo:hi]
            cursor += hi - lo
        return out


def load_cell_data(archive: Path) -> tuple[pd.DataFrame, list[str]]:
    with ZipFile(archive) as outer:
        with NestedZarr(outer, "cells.zarr.zip") as cells:
            summary = cells.read_array("cell_summary")
        with NestedZarr(outer, "cell_feature_matrix.zarr.zip") as matrix:
            attrs = matrix.attrs("cell_features")
            feature_names = [str(x) for x in attrs["feature_keys"]]
            feature_to_index = {name: i for i, name in enumerate(feature_names)}
            missing = sorted(set(MARKERS) - set(feature_names))
            if missing:
                raise ValueError(f"Required markers missing from Xenium panel: {missing}")
            indptr = matrix.read_array("cell_features/indptr")
            counts = np.zeros((summary.shape[0], len(MARKERS)), dtype=np.float32)
            for col, gene in enumerate(MARKERS):
                feature_index = feature_to_index[gene]
                start, stop = int(indptr[feature_index]), int(indptr[feature_index + 1])
                rows = matrix.read_1d_slice("cell_features/indices", start, stop)
                values = matrix.read_1d_slice("cell_features/data", start, stop)
                counts[rows.astype(np.int64), col] = values.astype(np.float32)

    frame = pd.DataFrame(
        {
            "cell_index": np.arange(summary.shape[0], dtype=np.int64),
            "x_um": summary[:, 0],
            "y_um": summary[:, 1],
        }
    )
    for col, gene in enumerate(MARKERS):
        frame[gene] = counts[:, col]
    frame["total_panel_counts"] = counts.sum(axis=1)
    denom = np.maximum(frame["total_panel_counts"].to_numpy(), 1.0)
    for gene in ("PTN", "CCL5"):
        frame[f"{gene}_log1p_cp10k"] = np.log1p(10000.0 * frame[gene].to_numpy() / denom)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_um", "y_um"]).reset_index(drop=True)
    return frame, feature_names


def add_subset_flags(frame: pd.DataFrame) -> None:
    frame["protective_macrophage"] = (frame["CD68"] > 0) & (frame["CD163"] > 0) & (frame["CXCL10"] > 0)
    frame["mycaf"] = (frame["ACTA2"] > 0) & (frame["TAGLN"] > 0)
    frame["vascular_element"] = (frame["CXCL12"] > 0) | (frame["RGS5"] > 0)
    frame["egfr_positive"] = frame["EGFR"] > 0
    frame["cxcl12_positive"] = frame["CXCL12"] > 0
    frame["rgs5_positive"] = frame["RGS5"] > 0


def ripley_cross_k(
    coords: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    radii: tuple[float, ...],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    valid = np.isfinite(coords).all(axis=1)
    coords = coords[valid]
    mask_a = mask_a[valid]
    mask_b = mask_b[valid]
    n = len(coords)
    a_idx = np.flatnonzero(mask_a)
    b_idx = np.flatnonzero(mask_b)
    area = max((coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min()), 1.0)
    tree = cKDTree(coords)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for radius in radii:
        neighbor_lists = tree.query_ball_point(coords[a_idx], radius)
        observed_counts = np.fromiter((np.count_nonzero(mask_b[nbr]) for nbr in neighbor_lists), dtype=float, count=len(neighbor_lists))
        if np.any(mask_a & mask_b):
            observed_counts -= np.asarray(mask_a[a_idx] & mask_b[a_idx], dtype=float)
        observed_mean = float(observed_counts.mean()) if len(observed_counts) else np.nan
        observed_k = area * observed_mean / max(len(b_idx), 1)

        null_means = []
        for _ in range(permutations):
            random_b = np.zeros(n, dtype=bool)
            random_b[rng.choice(n, size=len(b_idx), replace=False)] = True
            null_counts = np.fromiter((np.count_nonzero(random_b[nbr]) for nbr in neighbor_lists), dtype=float, count=len(neighbor_lists))
            null_means.append(float(null_counts.mean()) if len(null_counts) else np.nan)
        null_means_arr = np.asarray(null_means, dtype=float)
        null_mean = float(np.nanmean(null_means_arr))
        null_sd = float(np.nanstd(null_means_arr, ddof=1)) if len(null_means_arr) > 1 else np.nan
        z = (observed_mean - null_mean) / null_sd if np.isfinite(null_sd) and null_sd > 0 else np.nan
        p_greater = (1 + int(np.count_nonzero(null_means_arr >= observed_mean))) / (len(null_means_arr) + 1)
        rows.append(
            {
                "radius_um": radius,
                "n_a": len(a_idx),
                "n_b": len(b_idx),
                "observed_mean_neighbors": observed_mean,
                "null_mean_neighbors": null_mean,
                "null_sd_neighbors": null_sd,
                "observed_cross_K_uncorrected": observed_k,
                "null_cross_K_uncorrected": area * null_mean / max(len(b_idx), 1),
                "observed_to_null_ratio": observed_mean / null_mean if null_mean > 0 else np.nan,
                "permutation_z": z,
                "permutation_p_greater": p_greater,
                "permutations": permutations,
            }
        )
    return pd.DataFrame(rows)


def cluster_mask(coords: np.ndarray, mask: np.ndarray, eps_um: float, min_cells: int = 1) -> tuple[np.ndarray, int]:
    indices = np.flatnonzero(mask)
    labels = np.full(len(coords), -1, dtype=np.int32)
    if len(indices) == 0:
        return labels, 0
    local = DBSCAN(eps=eps_um, min_samples=min_cells, n_jobs=-1).fit_predict(coords[indices])
    labels[indices] = local
    return labels, int(local.max() + 1 if np.any(local >= 0) else 0)


def retain_mixed_cluster_cells(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(frame), dtype=bool)
    for label in np.unique(labels[labels >= 0]):
        members = labels == label
        if frame.loc[members, "protective_macrophage"].any() and frame.loc[members, "mycaf"].any():
            keep |= members
    return keep


def nearest_distance(coords: np.ndarray, source_mask: np.ndarray, query_mask: np.ndarray) -> np.ndarray:
    out = np.full(query_mask.sum(), np.nan, dtype=float)
    source = coords[source_mask]
    query = coords[query_mask]
    if len(source) and len(query):
        out[:] = cKDTree(source).query(query, k=1)[0]
    return out


def gradient_table(frame: pd.DataFrame, distance: np.ndarray, reference: str) -> pd.DataFrame:
    bins = np.array([0, 25, 50, 100, 200, 400, 800, 1600, np.inf], dtype=float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        selected = np.isfinite(distance) & (distance >= lo) & (distance < hi)
        rows.append(
            {
                "reference_niche": reference,
                "distance_low_um": lo,
                "distance_high_um": hi,
                "n_egfr_cells": int(selected.sum()),
                "mean_PTN_log1p_cp10k": float(frame.loc[frame["egfr_positive"], "PTN_log1p_cp10k"].to_numpy()[selected].mean()) if selected.any() else np.nan,
                "mean_CCL5_log1p_cp10k": float(frame.loc[frame["egfr_positive"], "CCL5_log1p_cp10k"].to_numpy()[selected].mean()) if selected.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_overlay(frame: pd.DataFrame, mixed_mask: np.ndarray, vascular_mask: np.ndarray, output: Path) -> None:
    rng = np.random.default_rng(7)
    n = len(frame)
    background = np.arange(n)
    if n > 90000:
        background = rng.choice(background, size=90000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, ligand in zip(axes, ("PTN", "CCL5")):
        ax.scatter(frame.loc[background, "x_um"], frame.loc[background, "y_um"], s=0.25, c="lightgray", alpha=0.18, linewidths=0)
        values = frame[f"{ligand}_log1p_cp10k"].to_numpy()
        sc = ax.scatter(frame["x_um"], frame["y_um"], c=values, s=0.7, cmap="viridis", alpha=0.45, linewidths=0)
        ax.scatter(frame.loc[mixed_mask, "x_um"], frame.loc[mixed_mask, "y_um"], s=5, facecolors="none", edgecolors="red", linewidths=0.35, label="mixed macrophage–myCAF")
        ax.scatter(frame.loc[vascular_mask, "x_um"], frame.loc[vascular_mask, "y_um"], s=4, facecolors="none", edgecolors="deepskyblue", linewidths=0.35, label="vascular niche")
        ax.scatter(frame.loc[frame["egfr_positive"], "x_um"], frame.loc[frame["egfr_positive"], "y_um"], s=1.5, c="white", alpha=0.35, linewidths=0, label="EGFR+ cells")
        ax.set_title(f"{ligand} log1p(CP10K)")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.invert_yaxis()
        ax.legend(loc="upper right", markerscale=2, fontsize=8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-eps-um", type=float, default=75.0)
    parser.add_argument("--permutations", type=int, default=99)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    frame, feature_names = load_cell_data(args.archive)
    add_subset_flags(frame)
    coords = frame[["x_um", "y_um"]].to_numpy(dtype=float)

    subset_rows = []
    for name in ("protective_macrophage", "mycaf", "vascular_element", "cxcl12_positive", "rgs5_positive", "egfr_positive"):
        subset_rows.append({"subset": name, "positive_cells": int(frame[name].sum()), "rule": {
            "protective_macrophage": "CD68>0 AND CD163>0 AND CXCL10>0",
            "mycaf": "ACTA2>0 AND TAGLN>0",
            "vascular_element": "CXCL12>0 OR RGS5>0",
            "cxcl12_positive": "CXCL12>0",
            "rgs5_positive": "RGS5>0",
            "egfr_positive": "EGFR>0",
        }[name]})
    pd.DataFrame(subset_rows).to_csv(output / "subset_counts.csv", index=False)

    macrophage = frame["protective_macrophage"].to_numpy()
    mycaf = frame["mycaf"].to_numpy()
    vascular = frame["vascular_element"].to_numpy()
    egfr = frame["egfr_positive"].to_numpy()
    radii = (25.0, 50.0, 100.0, 200.0, 400.0)
    ripley = ripley_cross_k(coords, macrophage, mycaf, radii, args.permutations, 11)
    ripley.to_csv(output / "ripley_cross_k_macrophage_vs_mycaf.csv", index=False)

    mixed_labels, mixed_cluster_count = cluster_mask(coords, macrophage | mycaf, args.cluster_eps_um)
    mixed_cells = retain_mixed_cluster_cells(frame, mixed_labels)
    vascular_labels, vascular_cluster_count = cluster_mask(coords, vascular, args.cluster_eps_um)
    vascular_cells = vascular_labels >= 0
    mixed_distance = nearest_distance(coords, mixed_cells, egfr)
    vascular_distance = nearest_distance(coords, vascular_cells, egfr)
    egfr_frame = frame.loc[egfr, ["cell_index", "x_um", "y_um", "EGFR", "PTN_log1p_cp10k", "CCL5_log1p_cp10k"]].copy()
    egfr_frame["distance_to_nearest_mixed_macrophage_myCAF_cell_um"] = mixed_distance
    egfr_frame["distance_to_nearest_vascular_niche_cell_um"] = vascular_distance
    egfr_frame.to_csv(output / "egfr_nearest_niche_distances.csv", index=False)

    gradient = pd.concat(
        [gradient_table(frame, mixed_distance, "mixed_macrophage_myCAF"), gradient_table(frame, vascular_distance, "vascular")],
        ignore_index=True,
    )
    gradient.to_csv(output / "egfr_ligand_gradients.csv", index=False)
    make_overlay(frame, mixed_cells, vascular_cells, output / "s2_bot_ligand_niche_overlay.png")

    summary = {
        "sample_id": "S2_Bot",
        "archive": str(args.archive.resolve()),
        "n_cells_with_coordinates": int(len(frame)),
        "n_features_in_panel": int(len(feature_names)),
        "cluster_eps_um": args.cluster_eps_um,
        "mixed_macrophage_myCAF_clusters": mixed_cluster_count,
        "mixed_macrophage_myCAF_cells": int(mixed_cells.sum()),
        "vascular_clusters": vascular_cluster_count,
        "vascular_cells_in_clusters": int(vascular_cells.sum()),
        "egfr_positive_cells": int(egfr.sum()),
        "mean_egfr_distance_to_mixed_um": float(np.nanmean(mixed_distance)) if np.isfinite(mixed_distance).any() else np.nan,
        "median_egfr_distance_to_mixed_um": float(np.nanmedian(mixed_distance)) if np.isfinite(mixed_distance).any() else np.nan,
        "mean_egfr_distance_to_vascular_um": float(np.nanmean(vascular_distance)) if np.isfinite(vascular_distance).any() else np.nan,
        "median_egfr_distance_to_vascular_um": float(np.nanmedian(vascular_distance)) if np.isfinite(vascular_distance).any() else np.nan,
        "marker_positive_threshold": "raw Xenium count > 0",
        "ripley_edge_correction": "none; rectangular bounding-window estimate",
        "ripley_null": f"{args.permutations} random label permutations preserving B count",
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
