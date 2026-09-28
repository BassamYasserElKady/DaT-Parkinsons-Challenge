#!/usr/bin/env python3
"""Combine mask-only ResNet and morphometric SVM probabilities without test leakage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resnet_probabilities(root: Path, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    base = root / "results/ants_registered_top025_middle_nonempty_mask_resnet18_cbam"
    with (base / "split.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    uids = [row["uid"] for row in split_rows]
    y = np.asarray([int(row["is_pathologic"]) for row in split_rows], dtype=int)
    split = np.asarray([row["split"] for row in split_rows])
    x = np.load(base / "binary_mask_windows.npy")
    model = ResNet18CBAM3D().to(device)
    checkpoint = torch.load(base / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)), batch_size=32, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch, _ in loader:
            logits = model(batch.to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return uids, y, np.asarray(scores, dtype=float), split


def metric_row(name: str, y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y, pred),
        "auroc": roc_auc_score(y, score),
        "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/mask_resnet_morphometric_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uids, y, resnet_score, split = resnet_probabilities(root, device)

    morph = pd.read_csv(root / "results/ants_registered_top025_morphometric_medium_gaussian_svm/morphometric_features.csv").sort_values("uid")
    if morph["uid"].astype(str).tolist() != sorted(uids):
        raise ValueError("UID ordering mismatch between morphometric features and ResNet split")
    morph_names = [
        "left_area", "left_perimeter", "left_circularity", "left_major_axis_length", "left_minor_axis_length", "left_eccentricity",
        "right_area", "right_perimeter", "right_circularity", "right_major_axis_length", "right_minor_axis_length", "right_eccentricity",
    ]
    X_morph = morph[morph_names].to_numpy(dtype=float)
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    morph_model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma=1.0 / 24.0, probability=True, random_state=args.random_state))
    morph_model.fit(X_morph[train], y[train])
    morph_score = morph_model.predict_proba(X_morph)[:, 1]

    thresholds = np.linspace(0.10, 0.90, 161)
    weights = np.linspace(0.0, 1.0, 41)
    candidates = []
    for weight in weights:
        score = weight * resnet_score[validation] + (1.0 - weight) * morph_score[validation]
        for threshold in thresholds:
            row = metric_row("candidate", y[validation], score, threshold)
            row["resnet_weight"] = weight
            row["morphometric_weight"] = 1.0 - weight
            candidates.append(row)
    best_balanced = max(candidates, key=lambda row: (row["balanced_accuracy"], row["specificity"], row["sensitivity"], -row["threshold"]))
    constrained = [row for row in candidates if row["specificity"] >= 0.80]
    best_specificity_constrained = max(constrained, key=lambda row: (row["sensitivity"], row["specificity"], row["balanced_accuracy"]))

    metrics = []
    for name, score in [("resnet_mask_only", resnet_score), ("morphometric_svm", morph_score)]:
        metrics.append({"split": "test", **metric_row(name, y[test], score[test], 0.5)})
    for label, chosen in [("ensemble_best_balanced", best_balanced), ("ensemble_sensitivity_at_validation_specificity_0.80", best_specificity_constrained)]:
        weight = float(chosen["resnet_weight"])
        threshold = float(chosen["threshold"])
        combined = weight * resnet_score + (1.0 - weight) * morph_score
        metrics.append({"split": "validation", **chosen})
        metrics.append({"split": "test", **metric_row(label, y[test], combined[test], threshold), "resnet_weight": weight, "morphometric_weight": 1.0 - weight})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in metrics for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    combined_best = float(best_balanced["resnet_weight"]) * resnet_score + (1.0 - float(best_balanced["resnet_weight"])) * morph_score
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "actual", "split", "resnet_score", "morphometric_score", "ensemble_score"])
        writer.writerows([[uid, int(label), part, float(rs), float(ms), float(cs)] for uid, label, part, rs, ms, cs in zip(uids, y, split, resnet_score, morph_score, combined_best)])
    with (output / "weight_threshold_search.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(candidates[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    report = output / "report.md"
    report.write_text(
        "# Mask ResNet + morphometric SVM ensemble\n\n"
        "ResNet probabilities and morphometric-SVM probabilities were combined as a weighted average. The weight and threshold were selected using validation scans only; the test set remained untouched until final evaluation.\n\n"
        f"- ResNet source: binary top-0.25% masks, device `{device}`\n"
        "- Morphometric source: 12 registered top-0.25% shape features\n"
        "- Morphometric classifier: standardized medium-Gaussian RBF SVM\n"
        "- Ensemble search: ResNet weight 0.00-1.00, threshold 0.10-0.90\n"
        f"- Best-balanced validation setting: ResNet weight {float(best_balanced['resnet_weight']):.2f}, morphometric weight {float(best_balanced['morphometric_weight']):.2f}, threshold {float(best_balanced['threshold']):.3f}\n"
        f"- Specificity-constrained validation setting: ResNet weight {float(best_specificity_constrained['resnet_weight']):.2f}, morphometric weight {float(best_specificity_constrained['morphometric_weight']):.2f}, threshold {float(best_specificity_constrained['threshold']):.3f}\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
