#!/usr/bin/env python3
"""Evaluate raw and calibrated EGFR spatial correlations for all samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


def corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1]), float(spearmanr(x[valid], y[valid]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--gene", default="EGFR")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/egfr_spatial_metrics_all_samples.csv"))
    args = parser.parse_args()
    root = args.root.resolve()

    manifest = pd.read_csv(args.manifest if args.manifest.is_absolute() else root / args.manifest)
    split_by_sample = dict(zip(manifest["sample_id"].astype(str), manifest["split"].astype(str)))
    available = sorted(p.name for p in (root / args.tables_dir).iterdir() if p.is_dir())
    sample_ids = args.sample_id or available
    missing = sorted(set(sample_ids) - set(available))
    if missing:
        raise ValueError(f"No processed tables found for: {missing}")

    checkpoint = torch.load(
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    genes = [str(gene) for gene in checkpoint["genes"]]
    if args.gene not in genes:
        raise ValueError(f"{args.gene} is not in the fitted adapter gene list")
    gene_index = genes.index(args.gene)
    with np.load(args.stats if args.stats.is_absolute() else root / args.stats, allow_pickle=False) as stats:
        stats_genes = [str(gene) for gene in stats["genes"].tolist()]
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
    if stats_genes != genes:
        raise ValueError("Checkpoint and development-normalization genes do not match")

    model = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    model.to(device).eval()
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)

    records: list[dict] = []
    pooled_raw: list[np.ndarray] = []
    pooled_calibrated: list[np.ndarray] = []
    pooled_target: list[np.ndarray] = []
    pooled_by_split: dict[str, dict[str, list[np.ndarray]]] = {}
    for sample_id in sample_ids:
        sample_dir = root / args.tables_dir / sample_id
        prediction_frame = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
        target_frame = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
        if not prediction_frame[["spot", "x", "y"]].equals(target_frame[["spot", "x", "y"]]):
            raise ValueError(f"Prediction and target coordinates do not match for {sample_id}")
        prediction = prediction_frame[genes].to_numpy(np.float32)
        target = target_frame[genes].to_numpy(np.float32)
        normalized = (prediction - prediction_mean) / prediction_std
        with torch.inference_mode():
            calibrated_normalized = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        calibrated = calibrated_normalized * target_std + target_mean
        raw_gene = prediction[:, gene_index]
        calibrated_gene = calibrated[:, gene_index]
        target_gene = target[:, gene_index]
        raw_pearson, raw_spearman = corr_pair(raw_gene, target_gene)
        calibrated_pearson, calibrated_spearman = corr_pair(calibrated_gene, target_gene)
        records.append(
            {
                "sample_id": sample_id,
                "split": split_by_sample.get(sample_id, "unknown"),
                "gene": args.gene,
                "tiles": len(prediction),
                "raw_spatial_pearson": raw_pearson,
                "raw_spatial_spearman": raw_spearman,
                "calibrated_spatial_pearson": calibrated_pearson,
                "calibrated_spatial_spearman": calibrated_spearman,
                "pearson_gain": calibrated_pearson - raw_pearson,
                "spearman_gain": calibrated_spearman - raw_spearman,
            }
        )
        pooled_raw.append(raw_gene)
        pooled_calibrated.append(calibrated_gene)
        pooled_target.append(target_gene)
        split = split_by_sample.get(sample_id, "unknown")
        split_arrays = pooled_by_split.setdefault(split, {"raw": [], "calibrated": [], "target": []})
        split_arrays["raw"].append(raw_gene)
        split_arrays["calibrated"].append(calibrated_gene)
        split_arrays["target"].append(target_gene)

    pooled_raw_array = np.concatenate(pooled_raw)
    pooled_calibrated_array = np.concatenate(pooled_calibrated)
    pooled_target_array = np.concatenate(pooled_target)
    raw_pearson, raw_spearman = corr_pair(pooled_raw_array, pooled_target_array)
    calibrated_pearson, calibrated_spearman = corr_pair(pooled_calibrated_array, pooled_target_array)
    records.append(
        {
            "sample_id": "POOLED_ALL_SAMPLES",
            "split": "pooled",
            "gene": args.gene,
            "tiles": len(pooled_target_array),
            "raw_spatial_pearson": raw_pearson,
            "raw_spatial_spearman": raw_spearman,
            "calibrated_spatial_pearson": calibrated_pearson,
            "calibrated_spatial_spearman": calibrated_spearman,
            "pearson_gain": calibrated_pearson - raw_pearson,
            "spearman_gain": calibrated_spearman - raw_spearman,
        }
    )
    for split_name, arrays in sorted(pooled_by_split.items()):
        split_raw = np.concatenate(arrays["raw"])
        split_calibrated = np.concatenate(arrays["calibrated"])
        split_target = np.concatenate(arrays["target"])
        split_raw_pearson, split_raw_spearman = corr_pair(split_raw, split_target)
        split_calibrated_pearson, split_calibrated_spearman = corr_pair(split_calibrated, split_target)
        records.append(
            {
                "sample_id": f"POOLED_{split_name.upper()}",
                "split": "pooled",
                "gene": args.gene,
                "tiles": len(split_target),
                "raw_spatial_pearson": split_raw_pearson,
                "raw_spatial_spearman": split_raw_spearman,
                "calibrated_spatial_pearson": split_calibrated_pearson,
                "calibrated_spatial_spearman": split_calibrated_spearman,
                "pearson_gain": split_calibrated_pearson - split_raw_pearson,
                "spearman_gain": split_calibrated_spearman - split_raw_spearman,
            }
        )
    result = pd.DataFrame(records)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    summary_path = output.with_suffix(".md")
    sample_result = result[result["sample_id"] != "POOLED_ALL_SAMPLES"]
    final_test = sample_result[sample_result["split"] == "final_test"]
    final_pooled = result[result["sample_id"] == "POOLED_FINAL_TEST"].iloc[0]
    summary = [
        f"# {args.gene} spatial correlation across samples",
        "",
        "Metrics compare tile-level raw DeepSpot-M or calibrated-adapter output against the aligned Xenium target. The adapter was not refit during this evaluation.",
        "",
        f"- Samples evaluated: `{len(sample_result)}`",
        f"- Final-test samples: `{len(final_test)}`",
        f"- Pooled tiles: `{len(pooled_target_array):,}`",
        f"- Pooled raw Pearson/Spearman: `{raw_pearson:.4f}` / `{raw_spearman:.4f}`",
        f"- Pooled calibrated Pearson/Spearman: `{calibrated_pearson:.4f}` / `{calibrated_spearman:.4f}`",
        f"- Pooled final-test calibrated Pearson/Spearman: `{final_pooled['calibrated_spatial_pearson']:.4f}` / `{final_pooled['calibrated_spatial_spearman']:.4f}`",
        "",
        "See the CSV for per-sample values and gains.",
    ]
    summary_path.write_text("\n".join(summary) + "\n")
    print(result.to_string(index=False))
    print(f"output={output}")
    print(f"summary={summary_path}")
    print(f"device={device}")


if __name__ == "__main__":
    main()
