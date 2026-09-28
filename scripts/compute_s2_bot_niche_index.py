#!/usr/bin/env python3
"""Compute the EGFR-cell dormant-versus-vascular niche index for S2-Bot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from analyze_s2_bot_niches import DEFAULT_ARCHIVE, DEFAULT_OUTPUT, add_subset_flags, load_cell_data


ROOT = Path(__file__).resolve().parents[1]


def nearest_distance(coords: np.ndarray, source_mask: np.ndarray, query_mask: np.ndarray) -> np.ndarray:
    source = coords[source_mask]
    query = coords[query_mask]
    if not len(source):
        return np.full(len(query), np.nan, dtype=float)
    return cKDTree(source).query(query, k=1)[0]


def make_ratio_overlay(frame: pd.DataFrame, ratio: np.ndarray, output: Path) -> None:
    rng = np.random.default_rng(7)
    all_indices = np.arange(len(frame))
    background = rng.choice(all_indices, size=min(90000, len(all_indices)), replace=False)
    egfr = frame["egfr_positive"].to_numpy()
    egfr_indices = np.flatnonzero(egfr)
    finite = np.isfinite(ratio)
    clipped = np.clip(np.log2(np.maximum(ratio, 1e-3)), -3, 3)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.scatter(frame.iloc[background]["x_um"], frame.iloc[background]["y_um"], s=0.25, c="lightgray", alpha=0.16, linewidths=0)
    selected = finite
    sc = ax.scatter(
        frame.iloc[egfr_indices[selected]]["x_um"],
        frame.iloc[egfr_indices[selected]]["y_um"],
        c=clipped[selected],
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
        s=5,
        alpha=0.68,
        linewidths=0,
    )
    ax.set_title("S2-Bot EGFR+ niche index\nlog2(Dvascular / Ddormant)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2 distance ratio; blue=dormant-proximal, red=vascular-proximal")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame, _ = load_cell_data(DEFAULT_ARCHIVE)
    add_subset_flags(frame)
    coords = frame[["x_um", "y_um"]].to_numpy(dtype=float)
    dormant = frame["protective_macrophage"].to_numpy() | frame["mycaf"].to_numpy()
    vascular = frame["vascular_element"].to_numpy()
    egfr = frame["egfr_positive"].to_numpy()

    dormant_distance = nearest_distance(coords, dormant, egfr)
    vascular_distance = nearest_distance(coords, vascular, egfr)
    ratio = np.divide(
        vascular_distance,
        dormant_distance,
        out=np.full_like(vascular_distance, np.nan),
        where=np.isfinite(vascular_distance) & np.isfinite(dormant_distance) & (dormant_distance > 0),
    )
    ratio[(dormant_distance == 0) & (vascular_distance > 0)] = np.inf
    ratio[(dormant_distance == 0) & (vascular_distance == 0)] = np.nan

    classifications = np.full(len(ratio), "intermediate", dtype=object)
    classifications[np.isfinite(ratio) & (ratio >= 2.0)] = "dormant_proximal"
    classifications[np.isfinite(ratio) & (ratio <= 0.5)] = "vascular_proximal"
    classifications[~np.isfinite(ratio) & (dormant_distance == 0)] = "dormant_overlap"

    out_dir = DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_path = out_dir / "egfr_nearest_niche_distances.csv"
    distances = pd.read_csv(existing_path)
    if len(distances) != int(egfr.sum()):
        raise ValueError("Existing EGFR distance table does not match the current raw archive")
    distances["distance_to_nearest_dormant_niche_cell_um"] = dormant_distance
    distances["niche_index_ratio_Dvascular_over_Ddormant"] = ratio
    distances["exploratory_state_classification"] = classifications
    distances.to_csv(existing_path, index=False)

    class_counts = pd.Series(classifications).value_counts().rename_axis("classification").reset_index(name="egfr_cells")
    class_counts.to_csv(out_dir / "niche_index_class_counts.csv", index=False)
    make_ratio_overlay(frame, ratio, out_dir / "s2_bot_niche_index_overlay.png")

    finite_ratio = ratio[np.isfinite(ratio)]
    summary = {
        "sample_id": "S2_Bot",
        "dormant_niche_rule": "(CD68>0 AND CD163>0 AND CXCL10>0) OR (ACTA2>0 AND TAGLN>0)",
        "vascular_niche_rule": "CXCL12>0 OR RGS5>0",
        "positive_threshold": "raw Xenium count > 0",
        "dormant_niche_cells": int(dormant.sum()),
        "vascular_niche_cells": int(vascular.sum()),
        "egfr_positive_cells": int(egfr.sum()),
        "median_Ddormant_um": float(np.nanmedian(dormant_distance)),
        "median_Dvascular_um": float(np.nanmedian(vascular_distance)),
        "median_ratio_Dvascular_over_Ddormant": float(np.nanmedian(finite_ratio)),
        "ratio_quantile_25": float(np.nanquantile(finite_ratio, 0.25)),
        "ratio_quantile_75": float(np.nanquantile(finite_ratio, 0.75)),
        "classification_thresholds": "dormant_proximal >=2; vascular_proximal <=0.5; otherwise intermediate",
        "interpretation": "exploratory proximity classes; not validated dormant/proliferative cell states",
    }
    (out_dir / "niche_index_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(class_counts.to_string(index=False))


if __name__ == "__main__":
    main()
