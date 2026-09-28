#!/usr/bin/env python3
"""Select 3D ResNet/DenseNet soft-voting weights on validation, then test once."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


def scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for x, _ in loader:
            result.extend(torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(result, dtype=float)


def calculate(name: str, truth: np.ndarray, score: np.ndarray, weight: float | None = None) -> dict[str, float | int | str]:
    prediction = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "model": name,
        "resnet_weight": "" if weight is None else float(weight),
        "densenet_weight": "" if weight is None else float(1.0 - weight),
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, prediction)),
        "auroc": float(roc_auc_score(truth, score)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/ants_registered_consecutive_resnet18_cbam"
    output = root / "results/ants_registered_3d_resnet_densenet_weight_sweep"
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((source / "split.csv").open(newline="", encoding="utf-8")))
    inputs = np.load(source / "input_windows.npy").astype(np.float32, copy=False)
    labels = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    split = np.asarray([row["split"] for row in rows])
    val_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")
    tensor_x = torch.from_numpy(inputs).unsqueeze(1)
    tensor_y = torch.from_numpy(labels)
    val_loader = DataLoader(TensorDataset(tensor_x[val_idx], tensor_y[val_idx]), batch_size=2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(TensorDataset(tensor_x[test_idx], tensor_y[test_idx]), batch_size=2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = ResNet18CBAM3D().to(device)
    dense = DenseNet121_3D().to(device)
    resnet.load_state_dict(torch.load(root / "results/ants_registered_consecutive_resnet18_cbam/best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    dense.load_state_dict(torch.load(root / "results/ants_registered_consecutive_densenet121_3d/best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    val_resnet = scores(resnet, val_loader, device)
    val_dense = scores(dense, val_loader, device)
    test_resnet = scores(resnet, test_loader, device)
    test_dense = scores(dense, test_loader, device)
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    sweep = []
    for weight in np.linspace(0.0, 1.0, 1001):
        combined = weight * val_resnet + (1.0 - weight) * val_dense
        sweep.append({"resnet_weight": float(weight), "densenet_weight": float(1.0 - weight), "validation_auroc": float(roc_auc_score(y_val, combined)), "validation_accuracy": float(accuracy_score(y_val, combined >= 0.5))})
    best = max(sweep, key=lambda row: (row["validation_auroc"], row["validation_accuracy"], -abs(row["resnet_weight"] - 0.5)))
    weight = float(best["resnet_weight"])
    test_combined = weight * test_resnet + (1.0 - weight) * test_dense
    result_rows = [
        calculate("3D ResNet-18 + CBAM", y_test, test_resnet),
        calculate("3D DenseNet-121", y_test, test_dense),
        calculate("Validation-selected weighted 3D ensemble", y_test, test_combined, weight),
    ]
    with (output / "weight_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "resnet_score", "densenet_score", "ensemble_score", "ensemble_predicted"])
        writer.writeheader()
        for pos, index in enumerate(test_idx):
            writer.writerow({"uid": rows[index]["uid"], "actual": int(y_test[pos]), "resnet_score": float(test_resnet[pos]), "densenet_score": float(test_dense[pos]), "ensemble_score": float(test_combined[pos]), "ensemble_predicted": int(test_combined[pos] >= 0.5)})
    report = output / "report.md"
    report.write_text(
        "# Validation-selected 3D ResNet-18 and 3D DenseNet-121 ensemble\n\n"
        "- Weight selection: 1,001-point sweep from 0.000 to 1.000, maximizing validation AUROC.\n"
        f"- Selected validation weights: ResNet-18 `{weight:.3f}`, DenseNet-121 `{1.0 - weight:.3f}`.\n"
        f"- Validation AUROC at selected weights: `{best['validation_auroc']:.4f}`.\n"
        "- The selected weights were applied once to the untouched 409-scan test set.\n\n"
        "## Held-out test\n\n"
        + "\n".join(f"- **{row['model']}**: accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}, AUROC={row['auroc']:.4f}" for row in result_rows)
        + "\n", encoding="utf-8",
    )
    print(f"selected_weights=resnet={weight:.3f},densenet={1.0 - weight:.3f}")
    print(f"validation_auroc={best['validation_auroc']:.4f}")
    for row in result_rows:
        print(f"{row['model']}: accuracy={row['accuracy']:.4f}, auroc={row['auroc']:.4f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
