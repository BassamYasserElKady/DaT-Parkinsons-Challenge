#!/usr/bin/env python3
"""Recompute the S2_Bot niche analysis using epithelial candidates, not EGFR alone.

This is a marker-defined *epithelial candidate* analysis.  It must not be
interpreted as a pathology-confirmed malignant-cell mask: a targeted Xenium
panel has no copy-number or histology-derived malignancy label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from analyze_s2_bot_niches import NestedZarr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT
    / "data/raw/10x/section_2_bot/"
    / "Human_Breast_Biomarkers_S2_Bot_xe_outs.zip"
)
DEFAULT_OUTPUT = ROOT / "results/s2_bot_epithelial_niche_analysis"

EPITHELIAL_MARKERS = ("EPCAM", "KRT19", "CD24", "CLDN4")
CONTAMINANT_MARKERS = ("CD68", "CD163", "ACTA2", "TAGLN", "PECAM1", "RGS5")
ANCHOR_MARKERS = ("CXCL10", "CXCL12")
BIOMARKERS = ("TACSTD2", "FOXA1")
SUBTYPE_MARKERS = ("PGR", "GATA3", "FOXA1", "AGR3", "ERBB2", "AR")
REQUIRED_MARKERS = tuple(
    dict.fromkeys(
        EPITHELIAL_MARKERS
        + CONTAMINANT_MARKERS
        + ANCHOR_MARKERS
        + BIOMARKERS
        + SUBTYPE_MARKERS
    )
)


def read_cells(archive: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read the selected marker vectors plus total measured-feature counts."""
    with ZipFile(archive) as outer:
        with NestedZarr(outer, "cells.zarr.zip") as cells:
            summary = cells.read_array("cell_summary")
            names = [str(value) for value in cells.attrs("cell_summary")["column_names"]]
        x_index = names.index("cell_centroid_x")
        y_index = names.index("cell_centroid_y")

        with NestedZarr(outer, "cell_feature_matrix.zarr.zip") as matrix:
            attrs = matrix.attrs("cell_features")
            feature_names = [str(value) for value in attrs["feature_keys"]]
            feature_to_index = {name: index for index, name in enumerate(feature_names)}
            missing = sorted(set(REQUIRED_MARKERS) - set(feature_names))
            if missing:
                raise ValueError(f"S2_Bot panel is missing required markers: {missing}")

            indptr = matrix.read_array("cell_features/indptr")
            n_cells = summary.shape[0]
            selected_counts: dict[str, np.ndarray] = {
                marker: np.zeros(n_cells, dtype=np.float32) for marker in REQUIRED_MARKERS
            }
            for marker, feature_index in feature_to_index.items():
                if marker not in selected_counts:
                    continue
                start, stop = int(indptr[feature_index]), int(indptr[feature_index + 1])
                rows = matrix.read_1d_slice("cell_features/indices", start, stop).astype(np.int64)
                values = matrix.read_1d_slice("cell_features/data", start, stop).astype(np.float32)
                selected_counts[marker][rows] = values

            # A streaming sum is used so the full sparse matrix is never held
            # in memory.  This includes all measured Xenium features, which is
            # the available panel-wide library-size denominator.
            total_counts = np.zeros(n_cells, dtype=np.float64)
            data_size = int(matrix.meta("cell_features/data")["shape"][0])
            block_size = 2_000_000
            for start in range(0, data_size, block_size):
                stop = min(start + block_size, data_size)
                rows = matrix.read_1d_slice("cell_features/indices", start, stop).astype(np.int64)
                values = matrix.read_1d_slice("cell_features/data", start, stop).astype(np.float64)
                total_counts += np.bincount(rows, weights=values, minlength=n_cells)

    frame = pd.DataFrame(
        {
            "cell_index": np.arange(summary.shape[0], dtype=np.int64),
            "x_um": summary[:, x_index],
            "y_um": summary[:, y_index],
            "total_measured_feature_counts": total_counts,
        }
    )
    for marker, values in selected_counts.items():
        frame[marker] = values
        frame[f"{marker}_log1p_cp10k"] = np.log1p(
            10_000.0 * values / np.maximum(total_counts, 1.0)
        )
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_um", "y_um"]).reset_index(drop=True), feature_names


def nearest_distance(coords: np.ndarray, anchors: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if not anchors.any():
        return np.full(targets.sum(), np.nan, dtype=float)
    return cKDTree(coords[anchors]).query(coords[targets], k=1)[0]


def make_plot(cells: pd.DataFrame, candidates: pd.DataFrame, output: Path) -> None:
    rng = np.random.default_rng(7)
    background = rng.choice(cells.index.to_numpy(), size=min(100_000, len(cells)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis in axes:
        axis.scatter(cells.loc[background, "x_um"], cells.loc[background, "y_um"], s=0.22, c="#d1d5db", alpha=0.25, linewidths=0)
        axis.set_xlabel("X (µm)")
        axis.set_ylabel("Y (µm)")
        axis.invert_yaxis()

    zone_colors = {
        "Dormant_Shielded": "#2166ac",
        "Intermediate": "#bdbdbd",
        "Vascular_Proliferative": "#b2182b",
    }
    for zone, color in zone_colors.items():
        selected = candidates["spatial_zone"] == zone
        axes[0].scatter(candidates.loc[selected, "x_um"], candidates.loc[selected, "y_um"], s=1.0, c=color, alpha=0.62, linewidths=0, label=zone.replace("_", " "))
    axes[0].set_title("Epithelial candidate spatial zones")
    axes[0].legend(markerscale=4, fontsize=8, loc="upper right")

    valid = np.isfinite(candidates["niche_index"])
    scatter = axes[1].scatter(
        candidates.loc[valid, "x_um"],
        candidates.loc[valid, "y_um"],
        c=candidates.loc[valid, "niche_index"],
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
        s=1.0,
        alpha=0.62,
        linewidths=0,
    )
    axes[1].set_title("Niche index: log2(D dormant / D vascular)")
    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("negative = dormant-anchor proximal")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def differential_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dormant = candidates["spatial_zone"] == "Dormant_Shielded"
    vascular = candidates["spatial_zone"] == "Vascular_Proliferative"
    for marker in BIOMARKERS:
        column = f"{marker}_log1p_cp10k"
        dormant_values = candidates.loc[dormant, column].to_numpy(dtype=float)
        vascular_values = candidates.loc[vascular, column].to_numpy(dtype=float)
        statistic, p_value = mannwhitneyu(dormant_values, vascular_values, alternative="two-sided")
        rows.append(
            {
                "gene": marker,
                "normalization": "log1p(CP10K), all measured Xenium features as denominator",
                "dormant_shielded_cells": len(dormant_values),
                "vascular_proliferative_cells": len(vascular_values),
                "dormant_mean": dormant_values.mean(),
                "vascular_mean": vascular_values.mean(),
                "dormant_median": np.median(dormant_values),
                "vascular_median": np.median(vascular_values),
                "mean_difference_dormant_minus_vascular": dormant_values.mean() - vascular_values.mean(),
                "mann_whitney_u": statistic,
                "two_sided_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contaminant-log1p-cp10k-threshold", type=float, default=1.0)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    cells, feature_names = read_cells(args.archive)
    epithelial = (cells.loc[:, EPITHELIAL_MARKERS] > 0).any(axis=1)
    contaminant = (cells[[f"{marker}_log1p_cp10k" for marker in CONTAMINANT_MARKERS]] > args.contaminant_log1p_cp10k_threshold).any(axis=1)
    cells["is_epithelial_candidate"] = epithelial & ~contaminant
    cells["is_contaminant_marker_high"] = contaminant
    cells["dormant_anchor"] = (
        (cells["CXCL10"] > 0) & ((cells["CD68"] > 0) | (cells["CD163"] > 0))
    ) | ((cells["ACTA2"] > 0) & (cells["TAGLN"] > 0))
    cells["vascular_anchor"] = (cells["CXCL12"] > 0) | (cells["RGS5"] > 0)

    candidates = cells.loc[cells["is_epithelial_candidate"]].copy()
    coords = cells[["x_um", "y_um"]].to_numpy(dtype=float)
    candidate_mask = cells["is_epithelial_candidate"].to_numpy()
    candidates["D_dormant_um"] = nearest_distance(coords, cells["dormant_anchor"].to_numpy(), candidate_mask)
    candidates["D_vascular_um"] = nearest_distance(coords, cells["vascular_anchor"].to_numpy(), candidate_mask)
    candidates["niche_index"] = np.log2(
        (candidates["D_dormant_um"].to_numpy() + 1e-6)
        / (candidates["D_vascular_um"].to_numpy() + 1e-6)
    )
    candidates["spatial_zone"] = np.select(
        [candidates["niche_index"] <= -1.0, candidates["niche_index"] >= 1.0],
        ["Dormant_Shielded", "Vascular_Proliferative"],
        default="Intermediate",
    )

    keep_columns = [
        "cell_index", "x_um", "y_um", "total_measured_feature_counts", "D_dormant_um", "D_vascular_um", "niche_index", "spatial_zone",
        *REQUIRED_MARKERS,
        *[f"{marker}_log1p_cp10k" for marker in REQUIRED_MARKERS],
    ]
    candidates.loc[:, keep_columns].to_parquet(output / "epithelial_candidate_cells.parquet", index=False)
    candidates.loc[:, keep_columns].to_csv(output / "epithelial_candidate_cells.csv", index=False)

    zone_counts = candidates["spatial_zone"].value_counts().rename_axis("spatial_zone").reset_index(name="epithelial_candidate_cells")
    zone_counts["fraction_of_epithelial_candidates"] = zone_counts["epithelial_candidate_cells"] / len(candidates)
    zone_counts.to_csv(output / "niche_zone_counts.csv", index=False)
    differential_summary(candidates).to_csv(output / "dormancy_biomarker_tests.csv", index=False)

    subtype = pd.DataFrame(
        {
            "gene": SUBTYPE_MARKERS,
            "mean_log1p_cp10k": [candidates[f"{marker}_log1p_cp10k"].mean() for marker in SUBTYPE_MARKERS],
            "median_log1p_cp10k": [candidates[f"{marker}_log1p_cp10k"].median() for marker in SUBTYPE_MARKERS],
            "fraction_raw_count_positive": [(candidates[marker] > 0).mean() for marker in SUBTYPE_MARKERS],
        }
    )
    subtype.to_csv(output / "epithelial_candidate_subtype_markers.csv", index=False)
    make_plot(cells, candidates, output / "s2_bot_epithelial_niche_index.png")

    summary = {
        "sample_id": "S2_Bot",
        "n_cells": int(len(cells)),
        "n_panel_features": int(len(feature_names)),
        "epithelial_candidate_rule": "(EPCAM>0 OR KRT19>0 OR CD24>0 OR CLDN4>0) AND no listed contaminant marker exceeds log1p(CP10K)>threshold",
        "contaminant_markers": list(CONTAMINANT_MARKERS),
        "contaminant_log1p_cp10k_threshold": args.contaminant_log1p_cp10k_threshold,
        "epithelial_candidate_cells": int(len(candidates)),
        "dormant_anchor_rule": "(CXCL10>0 AND (CD68>0 OR CD163>0)) OR (ACTA2>0 AND TAGLN>0)",
        "dormant_anchor_cells": int(cells["dormant_anchor"].sum()),
        "vascular_anchor_rule": "CXCL12>0 OR RGS5>0",
        "vascular_anchor_cells": int(cells["vascular_anchor"].sum()),
        "niche_index": "log2((D_dormant + 1e-6)/(D_vascular + 1e-6)); negative values are dormant-anchor proximal",
        "dormant_shielded_threshold": "niche_index <= -1",
        "vascular_proliferative_threshold": "niche_index >= 1",
        "persister_fraction_proxy_percent": float(100.0 * (candidates["spatial_zone"] == "Dormant_Shielded").mean()),
        "caveat": "Spatial proximity zones and epithelial marker positivity do not establish cellular dormancy or tumor malignancy; this is an exploratory, marker-defined analysis.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(differential_summary(candidates).to_string(index=False))


if __name__ == "__main__":
    main()
