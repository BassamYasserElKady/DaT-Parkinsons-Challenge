#!/usr/bin/env python3
"""Train torchvision MaxViT-tiny on the registered grayscale slice windows.

MaxViT-tiny is a 2D architecture. The 16 selected grayscale slices are therefore
passed as 16 input channels so one forward pass sees the complete window.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import maxvit_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
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


def build_model(in_channels: int) -> nn.Module:
    model = maxvit_t(weights=None)
    first_conv = model.stem[0][0]
    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError(f"unexpected MaxViT stem: {type(first_conv)!r}")
    model.stem[0][0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    classifier = model.classifier[-1]
    if not isinstance(classifier, nn.Linear):
        raise TypeError(f"unexpected MaxViT classifier: {type(classifier)!r}")
    model.classifier[-1] = nn.Linear(classifier.in_features, 2, bias=classifier.bias is not None)
    return model


def resize_inputs(x: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, image_size: int):
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = resize_inputs(batch_x.to(device, non_blocking=True), image_size)
            logits = model(batch_x)
            truth.extend(batch_y.numpy().tolist())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, predictions)),
        "auroc": float(roc_auc_score(truth, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, predictions, scores


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    root = args.root.resolve()
    source = root / "results/ants_registered_consecutive_resnet18_cbam"
    output = root / "results/ants_registered_consecutive_maxvit_tiny"
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

    tensor_x = torch.from_numpy(inputs.astype(np.float32, copy=False))
    tensor_y = torch.from_numpy(y)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(TensorDataset(tensor_x[train_idx], tensor_y[train_idx]), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(tensor_x[val_idx], tensor_y[val_idx]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(TensorDataset(tensor_x[test_idx], tensor_y[test_idx]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(int(inputs.shape[1])).to(device)
    with torch.no_grad():
        check = model(resize_inputs(tensor_x[:1].to(device), args.image_size))
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
            batch_x = resize_inputs(batch_x.to(device, non_blocking=True), args.image_size)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * batch_x.shape[0]
        val_metrics, _, _ = evaluate(model, val_loader, device, args.image_size)
        scheduler.step(val_metrics["auroc"])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_idx),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
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
    test_metrics, predictions, scores = evaluate(model, test_loader, device, args.image_size)
    torch.save(
        {"model_state_dict": model.state_dict(), "input_channels": int(inputs.shape[1]), "image_size": args.image_size, "args": vars(args), "best_val_auroc": best_val_auc},
        output / "best_model.pt",
    )

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
        "model": "MaxViT-tiny (16 grayscale slices as channels)",
        "pretrained": False,
        "regularization": "L2 via AdamW weight_decay",
        "weight_decay": args.weight_decay,
        "device": str(device),
        "n_scans": len(y),
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "patients_total": int(y.sum()),
        "healthy_total": int((y == 0).sum()),
        "patients_test": int(y[test_idx].sum()),
        "healthy_test": int((y[test_idx] == 0).sum()),
        "best_validation_auroc": float(best_val_auc),
        **test_metrics,
    }
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    report = output / "report.md"
    report.write_text(
        "# Registered consecutive-window MaxViT-tiny\n\n"
        "- Input: ANTs-registered grayscale intensity windows from the same 1,362-scan dataset and split as the 3D ResNet-18 experiment\n"
        "- Slice selection: top 0.15% hotspot scoring, leading 4% of nonempty slices discarded, contiguous window approximately 7/41 of remaining slices\n"
        f"- Input shape: `{inputs.shape[1]} x {inputs.shape[2]} x {inputs.shape[3]}` slices x height x width; resized to `{args.image_size} x {args.image_size}`\n"
        "- Model: torchvision MaxViT-tiny, a 2D model adapted to accept the selected slices as input channels; this is not a native 3D convolutional network\n"
        "- Pretraining: none (`weights=None`)\n"
        f"- Regularization: L2 weight decay via AdamW, `{args.weight_decay}`\n"
        f"- Split: same stratified split as the 3D ResNet-18 run; random_state={args.random_state}\n"
        f"- Device: `{device}`\n\n"
        "## Held-out test\n\n"
        + "\n".join(f"- **{key}:** {value:.4f}" if isinstance(value, float) else f"- **{key}:** {value}" for key, value in metrics.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"test_metrics={metrics}")


if __name__ == "__main__":
    main()
