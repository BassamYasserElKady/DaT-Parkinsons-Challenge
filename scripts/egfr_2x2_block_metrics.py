#!/usr/bin/env python3
"""Evaluate EGFR-associated genes after pooling neighboring tiles in 2x2 blocks.

Four adjacent 224-pixel tiles become one observation. The split is selected by
the manifest and is never mixed; this permits a development analysis or a
final held-out evaluation without changing the procedure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def aggregate_complete_2x2(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Aggregate only groups containing four distinct neighboring grid tiles."""
    work = frame[["x", "y", *value_columns]].copy()
    work["grid_x"] = (work["x"] / 224).round().astype(int)
    work["grid_y"] = (work["y"] / 224).round().astype(int)
    work["block_x"] = work["grid_x"] // 2
    work["block_y"] = work["grid_y"] // 2
    keys = ["block_x", "block_y"]
    summary = work.groupby(keys, sort=False).agg(
        n=("grid_x", "size"),
        nx=("grid_x", "nunique"),
        ny=("grid_y", "nunique"),
        min_x=("grid_x", "min"),
        max_x=("grid_x", "max"),
        min_y=("grid_y", "min"),
        max_y=("grid_y", "max"),
    )
    valid = summary[
        (summary["n"] == 4)
        & (summary["nx"] == 2)
        & (summary["ny"] == 2)
        & (summary["max_x"] - summary["min_x"] == 1)
        & (summary["max_y"] - summary["min_y"] == 1)
    ].index
    if len(valid) == 0:
        return pd.DataFrame(columns=[*keys, *value_columns])
    filtered = work.set_index(keys).loc[valid].reset_index()
    return filtered.groupby(keys, sort=False)[value_columns].mean().reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--split", choices=["development", "final_test"], default="development")
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_2x2_blocks"))
    args = parser.parse_args()
    root = args.root.resolve()

    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = pd.read_csv(manifest_path)
    sample_ids = manifest.loc[manifest["split"].eq(args.split), "sample_id"].astype(str).tolist()
    if not sample_ids:
        raise ValueError(f"no samples in manifest split {args.split!r}")

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    genes = [str(gene) for gene in checkpoint["genes"]]
    with np.load(args.stats if args.stats.is_absolute() else root / args.stats, allow_pickle=False) as stats:
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
        stats_genes = [str(gene) for gene in stats["genes"].tolist()]
    if genes != stats_genes:
        raise ValueError("checkpoint and normalization genes differ")

    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()

    rows: list[pd.DataFrame] = []
    per_sample_counts = []
    for sample_id in sample_ids:
        sample_dir = root / args.tables_dir / sample_id
        prediction = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
        target = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
        if not prediction[["spot", "x", "y"]].equals(target[["spot", "x", "y"]]):
            raise ValueError(f"coordinate mismatch in {sample_id}")
        prediction_values = prediction[genes].to_numpy(np.float32)
        normalized = (prediction_values - prediction_mean) / prediction_std
        with torch.inference_mode():
            calibrated_normalized = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        calibrated = calibrated_normalized * target_std + target_mean

        coordinates = prediction[["x", "y"]].reset_index(drop=True)
        calibrated_frame = pd.concat([coordinates, pd.DataFrame(calibrated, columns=genes)], axis=1)
        raw_frame = pd.concat([coordinates, prediction[genes].reset_index(drop=True)], axis=1)
        target_frame = pd.concat([coordinates, target[genes].reset_index(drop=True)], axis=1)
        cal_blocks = aggregate_complete_2x2(calibrated_frame, genes)
        raw_blocks = aggregate_complete_2x2(raw_frame, genes)
        target_blocks = aggregate_complete_2x2(target_frame, genes)
        if len(cal_blocks) != len(raw_blocks) or len(cal_blocks) != len(target_blocks):
            raise ValueError(f"2x2 block mismatch in {sample_id}")
        cal_blocks["sample_id"] = sample_id
        raw_blocks["sample_id"] = sample_id
        target_blocks["sample_id"] = sample_id
        rows.append(pd.concat([cal_blocks.add_prefix("calibrated_"), raw_blocks.add_prefix("raw_"), target_blocks.add_prefix("xenium_")], axis=1))
        per_sample_counts.append({"sample_id": sample_id, "complete_2x2_blocks": len(cal_blocks), "source_tiles": len(prediction)})

    blocks = pd.concat(rows, ignore_index=True)
    result_rows = []
    for gene in genes:
        xenium_egfr = blocks["xenium_EGFR"].to_numpy(float)
        calibrated_gene = blocks[f"calibrated_{gene}"].to_numpy(float)
        raw_gene = blocks[f"raw_{gene}"].to_numpy(float)
        xenium_gene = blocks[f"xenium_{gene}"].to_numpy(float)
        cal_p, cal_s = corr(calibrated_gene, xenium_egfr)
        raw_p, raw_s = corr(raw_gene, xenium_egfr)
        xenium_p, xenium_s = corr(xenium_gene, xenium_egfr)
        result_rows.append(
            {
                "gene": gene,
                "blocks": len(blocks),
                "calibrated_prediction_vs_xenium_egfr_pearson": cal_p,
                "calibrated_prediction_vs_xenium_egfr_spearman": cal_s,
                "raw_prediction_vs_xenium_egfr_pearson": raw_p,
                "raw_prediction_vs_xenium_egfr_spearman": raw_s,
                "xenium_gene_vs_xenium_egfr_pearson": xenium_p,
                "xenium_gene_vs_xenium_egfr_spearman": xenium_s,
            }
        )
    metrics = pd.DataFrame(result_rows)
    metrics["calibrated_rank_score"] = metrics[["calibrated_prediction_vs_xenium_egfr_pearson", "calibrated_prediction_vs_xenium_egfr_spearman"]].mean(axis=1)
    metrics = metrics.sort_values("calibrated_rank_score", ascending=False)

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "gene_metrics.csv", index=False)
    pd.DataFrame(per_sample_counts).to_csv(output_dir / "block_counts.csv", index=False)
    blocks.to_parquet(output_dir / "development_2x2_blocks.parquet", index=False)
    egfr_row = metrics.loc[metrics["gene"].eq("EGFR")].iloc[0]
    top = metrics.head(15)
    report = [
        f"# {args.split.replace('_', ' ').title()} 2x2 spatial-block EGFR analysis",
        "",
        f"Each observation is the mean of four adjacent 224-pixel tiles. Only complete 2x2 rectangles were retained. Evaluated manifest split: `{args.split}` ({len(sample_ids)} samples).",
        "",
        f"Complete 2x2 blocks: {len(blocks):,}",
        f"Device used for calibration inference: {device}",
        "",
        "## EGFR row",
        "",
        pd.DataFrame([egfr_row]).to_markdown(index=False),
        "",
        "## Top genes by calibrated-prediction correlation with Xenium EGFR",
        "",
        top.to_markdown(index=False),
        "",
        "The gene-to-EGFR correlations are associations across 2x2 blocks, not evidence that a gene causes EGFR expression. This is exploratory; the final-test values are held-out evaluation values, not model-selection values.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"complete_blocks={len(blocks):,}; device={device}; outputs={output_dir}")
    print(top[["gene", "calibrated_prediction_vs_xenium_egfr_pearson", "calibrated_prediction_vs_xenium_egfr_spearman", "xenium_gene_vs_xenium_egfr_pearson", "xenium_gene_vs_xenium_egfr_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
