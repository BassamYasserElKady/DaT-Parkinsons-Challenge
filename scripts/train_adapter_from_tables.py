#!/usr/bin/env python3
"""Train and evaluate the DeepSpot-only calibration adapter.

Inputs are the derived per-sample parquet tables produced by
build_pilot_tables.py. The final-test samples are never used for training,
validation, normalization, or model selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device
from spatial_pipeline.losses import calibration_loss
from spatial_pipeline.metrics import evaluate_per_sample_and_pooled
from spatial_pipeline.manifest import assert_sample_disjoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adapter_baseline"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pearson-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.15)
    parser.add_argument("--rank-pairs", type=int, default=3000)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--latent-width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def read_table(root: Path, tables_dir: Path, sample_id: str, genes: list[str]):
    sample_dir = resolve(root, tables_dir) / sample_id
    prediction_path = next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet"))
    target_path = next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet"))
    prediction = pd.read_parquet(prediction_path)
    target = pd.read_parquet(target_path)
    if not prediction[["spot", "x", "y"]].equals(target[["spot", "x", "y"]]):
        raise ValueError(f"{sample_id}: prediction/target coordinates do not match")
    if any(gene not in prediction.columns or gene not in target.columns for gene in genes):
        raise ValueError(f"{sample_id}: missing development-defined gene")
    return prediction[genes].to_numpy(np.float32), target[genes].to_numpy(np.float32)


def load_inputs(args):
    manifest = pd.read_csv(resolve(args.root, args.manifest))
    if set(manifest["split"]) != {"development", "final_test"}:
        raise ValueError("manifest must contain development and final_test rows")
    development = manifest.loc[manifest["split"] == "development", "sample_id"].astype(str).tolist()
    final_test = manifest.loc[manifest["split"] == "final_test", "sample_id"].astype(str).tolist()
    assert_sample_disjoint(development, final_test)
    with np.load(resolve(args.root, args.stats), allow_pickle=False) as data:
        genes = [str(gene) for gene in data["genes"].tolist()]
        prediction_mean = data["prediction_mean"].astype(np.float32)
        prediction_std = data["prediction_std"].astype(np.float32)
        target_mean = data["target_mean"].astype(np.float32)
        target_std = data["target_std"].astype(np.float32)
    if len(genes) != len(prediction_mean):
        raise ValueError("normalization statistics have inconsistent gene dimensions")
    samples = {}
    for sample_id in development + final_test:
        prediction, target = read_table(args.root, args.tables_dir, sample_id, genes)
        samples[sample_id] = {
            "x": (prediction - prediction_mean) / prediction_std,
            "y": (target - target_mean) / target_std,
            "target_raw": target,
        }
    stats = {
        "genes": genes,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    return development, final_test, samples, stats


def grouped_folds(sample_ids: list[str], n_folds: int, seed: int):
    if n_folds < 2 or n_folds > len(sample_ids):
        raise ValueError("fold count must be between 2 and the number of development samples")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(sorted(sample_ids), dtype=object)[rng.permutation(len(sample_ids))])
    result = []
    for validation in np.array_split(shuffled, n_folds):
        validation = [str(value) for value in validation]
        training = [sample_id for sample_id in sample_ids if sample_id not in validation]
        assert_sample_disjoint(training, validation)
        result.append((training, validation))
    return result


def fit_model(train_ids, samples, config, args, device, epochs=None):
    epochs = epochs or args.epochs
    x = np.concatenate([samples[sample_id]["x"] for sample_id in train_ids], axis=0)
    y = np.concatenate([samples[sample_id]["y"] for sample_id in train_ids], axis=0)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = DeepSpotCalibrationAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = calibration_loss(
                model(xb), yb,
                pearson_weight=args.pearson_weight,
                rank_weight=args.rank_weight,
                rank_pairs=args.rank_pairs,
            )
            loss["total"].backward()
            optimizer.step()
    return model


def predict_samples(model, sample_ids, samples, stats, device):
    model.eval()
    output = {}
    with torch.inference_mode():
        for sample_id in sample_ids:
            x = torch.from_numpy(samples[sample_id]["x"]).to(device)
            normalized_prediction = model(x).cpu().numpy()
            prediction = normalized_prediction * stats["target_std"] + stats["target_mean"]
            output[sample_id] = (prediction.astype(np.float32), samples[sample_id]["target_raw"])
    return output


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = default_device()
    development, final_test, samples, stats = load_inputs(args)
    config = AdapterConfig(
        genes=len(stats["genes"]),
        hidden_width=args.hidden_width,
        latent_width=args.latent_width,
        dropout=args.dropout,
    )

    fold_rows = []
    for fold, (train_ids, validation_ids) in enumerate(grouped_folds(development, args.folds, args.seed), start=1):
        model = fit_model(train_ids, samples, config, args, device)
        validation_predictions = predict_samples(model, validation_ids, samples, stats, device)
        evaluated = [
            (sample_id, validation_predictions[sample_id][0], validation_predictions[sample_id][1], stats["genes"])
            for sample_id in validation_ids
        ]
        metrics = evaluate_per_sample_and_pooled(evaluated)
        metrics.insert(0, "fold", fold)
        metrics["training_samples"] = ",".join(sorted(train_ids))
        metrics["validation_samples"] = ",".join(sorted(validation_ids))
        fold_rows.append(metrics)

    final_model = fit_model(development, samples, config, args, device)
    final_predictions = predict_samples(final_model, final_test, samples, stats, device)
    final_evaluated = [
        (sample_id, final_predictions[sample_id][0], final_predictions[sample_id][1], stats["genes"])
        for sample_id in final_test
    ]
    final_metrics = evaluate_per_sample_and_pooled(final_evaluated)

    output_dir = resolve(args.root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(fold_rows, ignore_index=True).to_csv(output_dir / "grouped_validation_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_test_metrics.csv", index=False)
    for sample_id, (prediction, _) in final_predictions.items():
        np.save(output_dir / f"{sample_id}.npy", prediction)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "adapter_config": vars(config),
            "genes": stats["genes"],
            "target_mean": stats["target_mean"],
            "target_std": stats["target_std"],
            "development_samples": development,
            "final_test_samples": final_test,
            "device": str(device),
        },
        output_dir / "adapter_final.pt",
    )
    print(f"device={device}")
    print(f"development samples={len(development)}; final-test samples={len(final_test)}; genes={len(stats['genes'])}")
    print("Grouped validation:")
    print(pd.concat(fold_rows, ignore_index=True).to_string(index=False))
    print("Final test:")
    print(final_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
