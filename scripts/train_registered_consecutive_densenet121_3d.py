#!/usr/bin/env python3
"""Train a true 3D DenseNet-121 on the registered grayscale slice windows."""

from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class _DenseLayer3D(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, 1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return torch.cat([x, new_features], dim=1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int, bn_size: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                (f"denselayer{i + 1}", _DenseLayer3D(num_input_features + i * growth_rate, growth_rate, bn_size))
                for i in range(num_layers)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Transition3D(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(num_input_features, num_output_features, 1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet121_3D(nn.Module):
    """DenseNet-121 block configuration with 3D operators."""

    def __init__(self, num_classes: int = 2, growth_rate: int = 32, block_config: tuple[int, ...] = (6, 12, 24, 16), num_init_features: int = 64) -> None:
        super().__init__()
        features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(1, num_init_features, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
                ]
            )
        )
        num_features = num_init_features
        for block_index, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers, num_features, growth_rate)
            features.add_module(f"denseblock{block_index + 1}", block)
            num_features += num_layers * growth_rate
            if block_index != len(block_config) - 1:
                transition = _Transition3D(num_features, num_features // 2)
                features.add_module(f"transition{block_index + 1}", transition)
                num_features //= 2
        features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features = features
        self.classifier = nn.Linear(num_features, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.relu(features)
        features = torch.nn.functional.adaptive_avg_pool3d(features, 1).flatten(1)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device, non_blocking=True))
            truth.extend(batch_y.numpy().tolist())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, predictions)),
        "auroc": float(roc_auc_score(truth, scores)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }, predictions, scores


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive")
    set_seed(args.random_state)
    root = args.root.resolve()
    source = root / "results/ants_registered_consecutive_resnet18_cbam"
    output = root / "results/ants_registered_consecutive_densenet121_3d"
    output.mkdir(parents=True, exist_ok=True)

    inputs = np.load(source / "input_windows.npy")
    if inputs.ndim != 4:
        raise ValueError(f"expected N,D,H,W input, got {inputs.shape}")
    split_rows = list(csv.DictReader((source / "split.csv").open(newline="", encoding="utf-8")))
    if len(split_rows) != len(inputs):
        raise ValueError(f"split rows ({len(split_rows)}) do not match inputs ({len(inputs)})")
    y = np.asarray([int(row["is_pathologic"]) for row in split_rows], dtype=np.int64)
    split = np.asarray([row["split"] for row in split_rows])
    uids = [row["uid"] for row in split_rows]
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")

    tensor_x = torch.from_numpy(inputs.astype(np.float32, copy=False)).unsqueeze(1)
    tensor_y = torch.from_numpy(y)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(TensorDataset(tensor_x[train_idx], tensor_y[train_idx]), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(tensor_x[val_idx], tensor_y[val_idx]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(TensorDataset(tensor_x[test_idx], tensor_y[test_idx]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121_3D().to(device)
    with torch.no_grad():
        check = model(tensor_x[:1].to(device))
    if tuple(check.shape) != (1, 2):
        raise RuntimeError(f"unexpected model output shape: {tuple(check.shape)}")
    class_counts = np.bincount(y[train_idx], minlength=2).astype(float)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * class_counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_val_auc = -np.inf
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * batch_x.shape[0]
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auroc"])
        row = {"epoch": epoch, "train_loss": running_loss / len(train_idx), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        print(f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} val_auc={val_metrics['auroc']:.4f} val_acc={val_metrics['accuracy']:.4f}", flush=True)
        if val_metrics["auroc"] > best_val_auc:
            best_val_auc = val_metrics["auroc"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("early_stopping=true", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, predictions, scores = evaluate(model, test_loader, device)
    torch.save({"model_state_dict": model.state_dict(), "input_shape": tuple(inputs.shape[1:]), "args": vars(args), "best_val_auroc": best_val_auc}, output / "best_model.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "predicted", "score_patient"])
        writer.writeheader()
        for position, index in enumerate(test_idx):
            writer.writerow({"uid": uids[index], "actual": int(y[index]), "predicted": int(predictions[position]), "score_patient": float(scores[position])})

    metrics = {
        "model": "3D DenseNet-121", "pretrained": False, "regularization": "L2 via AdamW weight_decay",
        "weight_decay": args.weight_decay, "device": str(device), "n_scans": len(y),
        "n_train": len(train_idx), "n_validation": len(val_idx), "n_test": len(test_idx),
        "patients_total": int(y.sum()), "healthy_total": int((y == 0).sum()),
        "patients_test": int(y[test_idx].sum()), "healthy_test": int((y[test_idx] == 0).sum()),
        "best_validation_auroc": float(best_val_auc), **test_metrics,
    }
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    report = output / "report.md"
    report.write_text(
        "# Registered consecutive-window 3D DenseNet-121\n\n"
        "- Input: ANTs-registered grayscale intensity windows from the same 1,362-scan dataset and split as the 3D ResNet-18 experiment\n"
        f"- Input shape: `{inputs.shape[1]} x {inputs.shape[2]} x {inputs.shape[3]}` depth x height x width\n"
        "- Model: DenseNet-121 block configuration implemented with 3D convolutions, 3D batch normalization, and 3D pooling\n"
        "- Pretraining: none; this 3D model was trained from scratch\n"
        f"- Regularization: L2 weight decay via AdamW, `{args.weight_decay}`\n"
        f"- Split: same stratified split as the 3D ResNet-18 run; random_state={args.random_state}\n"
        f"- Device: `{device}`\n\n## Held-out test\n\n"
        + "\n".join(f"- **{key}:** {value:.4f}" if isinstance(value, float) else f"- **{key}:** {value}" for key, value in metrics.items())
        + "\n", encoding="utf-8",
    )
    print(f"output={output}")
    print(f"test_metrics={metrics}")


if __name__ == "__main__":
    main()
