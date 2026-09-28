#!/usr/bin/env python3
"""Train an elastic-net EGFR-presence classifier from calibrated gene predictions.

The classifier is deliberately tile-level: labels are ``Xenium EGFR > 0`` in
the aligned tile target table.  The seven genes are fixed before the final-test
evaluation; no final-test labels are used for fitting or threshold selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device


DEFAULT_GENES = ["CD24", "TACSTD2", "SDC1", "ENO1", "KRT15", "LENG8", "CIRBP"]


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ordered = label[order].astype(np.float64)
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(score) + 1)
    return float((precision * ordered).sum() / positives)


def balanced_accuracy(score: np.ndarray, label: np.ndarray, threshold: float = 0.5) -> float:
    predicted = score >= threshold
    label = label.astype(bool)
    positive = label.sum()
    negative = (~label).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    sensitivity = (predicted & label).sum() / positive
    specificity = ((~predicted) & (~label)).sum() / negative
    return float((sensitivity + specificity) / 2)


def fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    l1: float,
    l2: float,
) -> tuple[nn.Linear, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    z = torch.from_numpy((x - mean) / scale).to(device)
    target = torch.from_numpy(y.astype(np.float32)).to(device)
    model = nn.Linear(x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positive = float(y.sum())
    negative = float(len(y) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(z).squeeze(1)
        loss = loss_fn(logits, target)
        loss = loss + l1 * model.weight.abs().mean() + l2 * model.weight.square().mean()
        loss.backward()
        optimizer.step()
    return model, mean, scale


def predict_classifier(model: nn.Linear, mean: np.ndarray, scale: np.ndarray, x: np.ndarray, device: torch.device) -> np.ndarray:
    z = torch.from_numpy(((x - mean) / scale).astype(np.float32)).to(device)
    with torch.inference_mode():
        return torch.sigmoid(model(z).squeeze(1)).cpu().numpy()


def load_calibrated_sample(
    root: Path,
    tables_dir: Path,
    sample_id: str,
    genes: list[str],
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    adapter: DeepSpotCalibrationAdapter,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sample_dir = resolve(root, tables_dir) / sample_id
    prediction_frame = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
    target_frame = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
    keys = ["spot", "x", "y"]
    if not prediction_frame[keys].equals(target_frame[keys]):
        raise ValueError(f"prediction/target coordinates do not match for {sample_id}")
    raw = prediction_frame[genes].to_numpy(np.float32)
    normalized = (raw - prediction_mean) / prediction_std
    with torch.inference_mode():
        calibrated_normalized = adapter(torch.from_numpy(normalized).to(device)).cpu().numpy()
    calibrated = calibrated_normalized * target_std + target_mean
    features = calibrated[:, [genes.index(g) for g in DEFAULT_GENES]].astype(np.float32)
    labels = (target_frame["EGFR"].to_numpy(np.float32) > 0).astype(np.float32)
    coordinates = target_frame[keys].copy()
    coordinates.insert(0, "sample_id", sample_id)
    coordinates["egfr_positive"] = labels.astype(bool)
    return features, labels, coordinates


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "tiles": int(len(labels)),
        "egfr_positive_fraction": float(labels.mean()),
        "auroc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "balanced_accuracy_at_0.5": balanced_accuracy(scores, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_candidate_classifier"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l1", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = args.root.resolve()
    manifest = pd.read_csv(resolve(root, args.manifest))
    development = manifest.loc[manifest["split"].eq("development"), "sample_id"].astype(str).tolist()
    final_test = manifest.loc[manifest["split"].eq("final_test"), "sample_id"].astype(str).tolist()

    checkpoint = torch.load(resolve(root, args.checkpoint), map_location="cpu", weights_only=False)
    adapter_genes = [str(g) for g in checkpoint["genes"]]
    missing = [gene for gene in DEFAULT_GENES if gene not in adapter_genes]
    if missing:
        raise ValueError(f"candidate genes absent from adapter: {missing}")
    with np.load(resolve(root, args.stats), allow_pickle=False) as stats:
        stats_genes = [str(g) for g in stats["genes"].tolist()]
        prediction_mean = stats["prediction_mean"].astype(np.float32)
        prediction_std = stats["prediction_std"].astype(np.float32)
    if stats_genes != adapter_genes:
        raise ValueError("adapter checkpoint and normalization genes differ")
    adapter = DeepSpotCalibrationAdapter(AdapterConfig(**checkpoint["adapter_config"]))
    adapter.load_state_dict(checkpoint["model_state_dict"])
    device = default_device()
    adapter.to(device).eval()
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)

    all_ids = development + final_test
    samples = {}
    for sample_id in all_ids:
        samples[sample_id] = load_calibrated_sample(
            root, args.tables_dir, sample_id, adapter_genes, prediction_mean, prediction_std,
            adapter, target_mean, target_std, device,
        )

    rng = np.random.default_rng(args.seed)
    shuffled = list(np.asarray(sorted(development), dtype=object)[rng.permutation(len(development))])
    folds = np.array_split(shuffled, args.folds)
    validation_rows: list[dict[str, object]] = []
    for fold_index, validation_array in enumerate(folds, 1):
        validation_ids = [str(v) for v in validation_array]
        train_ids = [sample_id for sample_id in development if sample_id not in validation_ids]
        train_x = np.concatenate([samples[sid][0] for sid in train_ids])
        train_y = np.concatenate([samples[sid][1] for sid in train_ids])
        val_x = np.concatenate([samples[sid][0] for sid in validation_ids])
        val_y = np.concatenate([samples[sid][1] for sid in validation_ids])
        model, mean, scale = fit_classifier(train_x, train_y, device, args.epochs, args.learning_rate, args.l1, args.l2)
        val_score = predict_classifier(model, mean, scale, val_x, device)
        metrics = evaluate(val_score, val_y)
        metrics.update({"fold": fold_index, "validation_samples": ",".join(sorted(validation_ids))})
        validation_rows.append(metrics)

    train_x = np.concatenate([samples[sid][0] for sid in development])
    train_y = np.concatenate([samples[sid][1] for sid in development])
    final_model, final_mean, final_scale = fit_classifier(train_x, train_y, device, args.epochs, args.learning_rate, args.l1, args.l2)

    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    final_metrics_rows: list[dict[str, object]] = []
    all_test_scores = []
    all_test_labels = []
    for sample_id in final_test:
        features, labels, coordinates = samples[sample_id]
        scores = predict_classifier(final_model, final_mean, final_scale, features, device)
        frame = coordinates.copy()
        frame["egfr_probability"] = scores
        prediction_rows.append(frame)
        sample_metrics = evaluate(scores, labels)
        sample_metrics["sample_id"] = sample_id
        final_metrics_rows.append(sample_metrics)
        all_test_scores.append(scores)
        all_test_labels.append(labels)
    pooled_scores = np.concatenate(all_test_scores)
    pooled_labels = np.concatenate(all_test_labels)
    pooled_metrics = evaluate(pooled_scores, pooled_labels)
    pooled_metrics["sample_id"] = "POOLED_FINAL_TEST"
    final_metrics_rows.append(pooled_metrics)

    validation = pd.DataFrame(validation_rows)
    final_metrics = pd.DataFrame(final_metrics_rows)
    validation.to_csv(output_dir / "grouped_development_validation_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_test_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(output_dir / "final_test_predictions.parquet", index=False)

    weights = final_model.weight.detach().cpu().numpy().ravel()
    coefficients = pd.DataFrame({"gene": DEFAULT_GENES, "standardized_logit_coefficient": weights})
    coefficients["absolute_coefficient"] = coefficients["standardized_logit_coefficient"].abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(output_dir / "final_model_coefficients.csv", index=False)
    (output_dir / "candidate_genes.txt").write_text("\n".join(DEFAULT_GENES) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "candidate_genes": DEFAULT_GENES,
            "feature_mean": final_mean,
            "feature_scale": final_scale,
            "development_samples": development,
            "final_test_samples": final_test,
            "label_definition": "aligned Xenium EGFR log1p(CP10K) > 0 at tile level",
            "elastic_net": {"l1": args.l1, "l2": args.l2},
        },
        output_dir / "egfr_candidate_classifier_final.pt",
    )

    validation_summary = validation[["auroc", "average_precision", "balanced_accuracy_at_0.5"]].mean()
    report = [
        "# EGFR candidate-gene classifier",
        "",
        "This is a tile-level classifier. A positive label means the aligned Xenium tile has EGFR > 0; it is not yet an individual-cell classifier.",
        "",
        f"Candidate genes: {', '.join(DEFAULT_GENES)}",
        f"Development samples: {', '.join(development)}",
        f"Final-test samples: {', '.join(final_test)}",
        f"Adapter device: {device}",
        "",
        "## Grouped development validation mean",
        "",
        validation_summary.to_frame("mean_metric").to_markdown(),
        "",
        "## Final-test metrics",
        "",
        final_metrics.to_markdown(index=False),
        "",
        "The final-test samples were not used to fit the classifier, choose the threshold, or alter the candidate-gene list.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Candidate genes:", ", ".join(DEFAULT_GENES))
    print("Development validation:")
    print(validation.to_string(index=False))
    print("Final test:")
    print(final_metrics.to_string(index=False))
    print("Coefficients:")
    print(coefficients.sort_values("absolute_coefficient", ascending=False).to_string(index=False))
    print(f"outputs={output_dir}")


if __name__ == "__main__":
    main()
