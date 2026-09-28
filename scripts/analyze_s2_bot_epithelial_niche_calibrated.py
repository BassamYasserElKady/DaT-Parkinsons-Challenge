#!/usr/bin/env python3
"""S2_Bot epithelial-niche analysis using calibrated DeepSpotM tiles.

Calibrated DeepSpotM predictions supply all marker expression values and all
spatial classifications. Xenium tile targets are used only to select marker-
specific cutoffs corresponding to log1p(CP10K) > 1 detection in the earlier
cell-level analysis. This avoids treating the adapter's smooth nonzero output
as literal single-molecule detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


DEFAULT_SAMPLE = "S2_Bot"
DEFAULT_OUTPUT = ROOT / "results/s2_bot_epithelial_niche_calibrated_deepspotm"
EPITHELIAL_MARKERS = ("EPCAM", "KRT19", "CD24", "CLDN4")
CONTAMINANT_MARKERS = ("CD68", "CD163", "ACTA2", "TAGLN", "PECAM1", "RGS5")
BIOMARKERS = ("TACSTD2", "FOXA1")
SUBTYPE_MARKERS = ("PGR", "GATA3", "FOXA1", "AGR3", "ERBB2", "AR")
ANCHOR_MARKERS = ("CXCL10", "CXCL12", "CD68", "CD163", "ACTA2", "TAGLN", "RGS5")
REQUIRED_MARKERS = tuple(dict.fromkeys(
    EPITHELIAL_MARKERS + CONTAMINANT_MARKERS + BIOMARKERS + SUBTYPE_MARKERS + ANCHOR_MARKERS
))


def f1_detection_threshold(score: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Choose a score cutoff that best matches target > 1 log1p(CP10K)."""
    positive = np.asarray(target > 1.0, dtype=bool)
    order = np.argsort(-score, kind="mergesort")
    sorted_positive = positive[order].astype(np.int64)
    tp = np.cumsum(sorted_positive)
    fp = np.cumsum(1 - sorted_positive)
    fn = int(positive.sum()) - tp
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
    best = int(np.argmax(f1))
    return float(score[order[best]]), float(f1[best])


def load_calibrated_tiles(root: Path, sample: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    sample_dir = root / "results/deepspot_tables" / sample
    prediction_path = next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet"))
    target_path = next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet"))
    prediction_frame = pd.read_parquet(prediction_path)
    target_frame = pd.read_parquet(target_path)
    coordinates = ["spot", "x", "y"]
    if not prediction_frame[coordinates].equals(target_frame[coordinates]):
        raise ValueError("DeepSpotM and Xenium tile coordinates do not match")

    checkpoint = torch.load(root / "results/adapter_baseline/adapter_final.pt", map_location="cpu", weights_only=False)
    stats = np.load(root / "results/deepspot_tables/development_normalization.npz", allow_pickle=False)
    genes = [str(gene) for gene in stats["genes"].tolist()]
    prediction_mean = stats["prediction_mean"].astype(np.float32)
    prediction_std = stats["prediction_std"].astype(np.float32)
    stats.close()
    if genes != [str(gene) for gene in checkpoint["genes"]]:
        raise ValueError("adapter checkpoint and development statistics genes do not match")
    missing = sorted(set(REQUIRED_MARKERS) - set(genes))
    if missing:
        raise ValueError(f"calibrated DeepSpotM output is missing markers: {missing}")

    prediction = prediction_frame[genes].to_numpy(np.float32)
    normalized = (prediction - prediction_mean) / prediction_std
    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()
    with torch.inference_mode():
        calibrated_normalized = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
    calibrated = calibrated_normalized * np.asarray(checkpoint["target_std"], dtype=np.float32) + np.asarray(checkpoint["target_mean"], dtype=np.float32)
    calibrated_frame = prediction_frame[coordinates].copy()
    calibrated_frame.loc[:, genes] = calibrated
    target_frame = target_frame[coordinates + genes].copy()
    return calibrated_frame, target_frame, genes, str(device)


def nearest_distance(coords: np.ndarray, anchors: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if not anchors.any():
        return np.full(int(targets.sum()), np.nan, dtype=float)
    return cKDTree(coords[anchors]).query(coords[targets], k=1)[0]


def make_plot(tiles: pd.DataFrame, candidates: pd.DataFrame, output: Path, mask_mode: str) -> None:
    rng = np.random.default_rng(7)
    background = rng.choice(tiles.index.to_numpy(), size=min(12_000, len(tiles)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for axis in axes:
        axis.scatter(tiles.loc[background, "x"], tiles.loc[background, "y"], s=1.0, c="#d1d5db", alpha=0.25, linewidths=0)
        axis.set_xlabel("DeepSpotM tile x (px)")
        axis.set_ylabel("DeepSpotM tile y (px)")
        axis.invert_yaxis()

    zone_colors = {"Dormant_Shielded": "#2166ac", "Intermediate": "#bdbdbd", "Vascular_Proliferative": "#b2182b"}
    for zone, color in zone_colors.items():
        selected = candidates["spatial_zone"] == zone
        axes[0].scatter(candidates.loc[selected, "x"], candidates.loc[selected, "y"], s=4.0, c=color, alpha=0.7, linewidths=0, label=zone.replace("_", " "))
    axes[0].set_title(f"Calibrated DeepSpotM epithelial zones ({mask_mode} mask)")
    axes[0].legend(markerscale=2.5, fontsize=8, loc="upper right")

    valid = np.isfinite(candidates["niche_index"])
    scatter = axes[1].scatter(candidates.loc[valid, "x"], candidates.loc[valid, "y"], c=candidates.loc[valid, "niche_index"], cmap="coolwarm", vmin=-3, vmax=3, s=4.0, alpha=0.7, linewidths=0)
    axes[1].set_title("Niche index: log2(D dormant / D vascular)")
    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("negative = dormant-anchor proximal")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-mode", choices=("strict", "relaxed"), default="strict")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    tiles, targets, genes, device = load_calibrated_tiles(ROOT, args.sample)
    thresholds = []
    for gene in REQUIRED_MARKERS:
        cutoff, f1 = f1_detection_threshold(tiles[gene].to_numpy(float), targets[gene].to_numpy(float))
        thresholds.append({"gene": gene, "calibrated_detection_cutoff": cutoff, "threshold_reference": "Xenium tile log1p(CP10K)>1", "threshold_f1": f1})
    threshold_frame = pd.DataFrame(thresholds).set_index("gene")
    positive = {gene: tiles[gene].to_numpy(float) >= threshold_frame.loc[gene, "calibrated_detection_cutoff"] for gene in REQUIRED_MARKERS}

    epithelial = np.logical_or.reduce([positive[gene] for gene in EPITHELIAL_MARKERS])
    contaminant = np.logical_or.reduce([positive[gene] for gene in CONTAMINANT_MARKERS])
    epithelial_strength = np.max(np.column_stack([tiles[gene].to_numpy(float) / threshold_frame.loc[gene, "calibrated_detection_cutoff"] for gene in EPITHELIAL_MARKERS]), axis=1)
    contaminant_strength = np.max(np.column_stack([tiles[gene].to_numpy(float) / threshold_frame.loc[gene, "calibrated_detection_cutoff"] for gene in CONTAMINANT_MARKERS]), axis=1)
    dormant_anchor = (positive["CXCL10"] & (positive["CD68"] | positive["CD163"])) | (positive["ACTA2"] & positive["TAGLN"])
    vascular_anchor = positive["CXCL12"] | positive["RGS5"]
    if args.mask_mode == "relaxed":
        candidate_mask = (epithelial_strength >= 1.0) & (contaminant_strength < 1.5 * epithelial_strength)
        candidate_rule = "max epithelial marker / cutoff >= 1 and max contaminant marker / cutoff < 1.5 * epithelial strength"
    else:
        candidate_mask = epithelial & ~contaminant
        candidate_rule = "(EPCAM OR KRT19 OR CD24 OR CLDN4 calibrated-positive) AND no listed contaminant marker calibrated-positive"
    tiles["epithelial_strength"] = epithelial_strength
    tiles["contaminant_strength"] = contaminant_strength
    tiles["is_epithelial_candidate"] = candidate_mask
    tiles["is_contaminant_marker_high"] = contaminant
    tiles["dormant_anchor"] = dormant_anchor
    tiles["vascular_anchor"] = vascular_anchor

    candidates = tiles.loc[tiles["is_epithelial_candidate"]].copy()
    coords = tiles[["x", "y"]].to_numpy(float)
    candidate_mask = tiles["is_epithelial_candidate"].to_numpy(bool)
    candidates["D_dormant_px"] = nearest_distance(coords, dormant_anchor, candidate_mask)
    candidates["D_vascular_px"] = nearest_distance(coords, vascular_anchor, candidate_mask)
    candidates["niche_index"] = np.log2((candidates["D_dormant_px"] + 1e-6) / (candidates["D_vascular_px"] + 1e-6))
    candidates["spatial_zone"] = np.select([candidates["niche_index"] <= -1.0, candidates["niche_index"] >= 1.0], ["Dormant_Shielded", "Vascular_Proliferative"], default="Intermediate")

    selected_columns = ["spot", "x", "y", "is_epithelial_candidate", "is_contaminant_marker_high", "dormant_anchor", "vascular_anchor", "D_dormant_px", "D_vascular_px", "niche_index", "spatial_zone", *REQUIRED_MARKERS]
    candidates[selected_columns].to_parquet(output / "calibrated_epithelial_candidate_tiles.parquet", index=False)
    candidates[selected_columns].to_csv(output / "calibrated_epithelial_candidate_tiles.csv", index=False)
    tiles[["spot", "x", "y", *REQUIRED_MARKERS]].to_parquet(output / "calibrated_marker_tiles.parquet", index=False)
    threshold_frame.to_csv(output / "calibrated_detection_thresholds.csv")

    zone_counts = candidates["spatial_zone"].value_counts().rename_axis("spatial_zone").reset_index(name="epithelial_candidate_tiles")
    zone_counts["fraction_of_epithelial_candidates"] = zone_counts["epithelial_candidate_tiles"] / max(len(candidates), 1)
    zone_counts.to_csv(output / "niche_zone_counts.csv", index=False)

    dormant = candidates["spatial_zone"] == "Dormant_Shielded"
    vascular = candidates["spatial_zone"] == "Vascular_Proliferative"
    test_rows = []
    for gene in BIOMARKERS:
        d = candidates.loc[dormant, gene].to_numpy(float)
        v = candidates.loc[vascular, gene].to_numpy(float)
        u, p = mannwhitneyu(d, v, alternative="two-sided")
        test_rows.append({"gene": gene, "expression_source": "calibrated DeepSpotM", "dormant_shielded_tiles": len(d), "vascular_proliferative_tiles": len(v), "dormant_mean": d.mean(), "vascular_mean": v.mean(), "dormant_median": np.median(d), "vascular_median": np.median(v), "mean_difference_dormant_minus_vascular": d.mean() - v.mean(), "mann_whitney_u": u, "two_sided_p_value": p})
    pd.DataFrame(test_rows).to_csv(output / "dormancy_biomarker_tests.csv", index=False)

    subtype = pd.DataFrame({"gene": SUBTYPE_MARKERS, "expression_source": "calibrated DeepSpotM", "mean_calibrated_log1p_cp10k": [candidates[g].mean() for g in SUBTYPE_MARKERS], "median_calibrated_log1p_cp10k": [candidates[g].median() for g in SUBTYPE_MARKERS], "fraction_above_calibrated_detection_cutoff": [(candidates[g] >= threshold_frame.loc[g, "calibrated_detection_cutoff"]).mean() for g in SUBTYPE_MARKERS]})
    subtype.to_csv(output / "epithelial_candidate_subtype_markers.csv", index=False)
    make_plot(tiles, candidates, output / "s2_bot_epithelial_niche_calibrated_deepspotm.png", args.mask_mode)

    summary = {"sample_id": args.sample, "expression_source": "calibrated DeepSpotM tile predictions", "mask_mode": args.mask_mode, "n_tiles": int(len(tiles)), "epithelial_candidate_tiles": int(len(candidates)), "detection_threshold_reference": "gene-specific calibrated score cutoff selected against Xenium tile log1p(CP10K)>1; Xenium values are not used for final expression measurements or spatial distances", "epithelial_candidate_rule": candidate_rule, "dormant_anchor_rule": "CXCL10 AND (CD68 OR CD163), OR ACTA2 AND TAGLN", "vascular_anchor_rule": "CXCL12 OR RGS5", "dormant_anchor_tiles": int(dormant_anchor.sum()), "vascular_anchor_tiles": int(vascular_anchor.sum()), "niche_index": "log2((D_dormant + 1e-6)/(D_vascular + 1e-6)); distances are DeepSpotM tile pixels", "dormant_shielded_threshold": "niche_index <= -1", "vascular_proliferative_threshold": "niche_index >= 1", "persister_fraction_proxy_percent": float(100.0 * dormant.mean()) if len(candidates) else None, "device": device, "caveat": "This is an exploratory tile-level proxy. Calibrated DeepSpotM smoothing is not equivalent to single-cell molecule detection and does not establish malignancy or dormancy."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    main()
