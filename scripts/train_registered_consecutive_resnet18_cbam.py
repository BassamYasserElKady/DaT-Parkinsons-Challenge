#!/usr/bin/env python3
"""Train a 3D ResNet-18 with CBAM on registered adaptive slice windows."""

from __future__ import annotations

import argparse
import csv
import copy
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-depth", type=int, default=16)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    return parser.parse_args()


def choose_window(data: np.ndarray) -> dict[str, int | float]:
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0).astype(np.float32, copy=False)
    threshold = float(np.percentile(data, 99.85))
    hot = data > threshold
    n_slices = int(data.shape[2])
    nonempty = np.flatnonzero(np.max(data, axis=(0, 1)) > 0)
    if nonempty.size == 0:
        raise ValueError("scan has no nonempty slices")
    discarded = min(int(ceil(0.04 * nonempty.size)), nonempty.size - 1)
    eligible = nonempty[discarded:]
    window = max(1, min(int(eligible.size), int(round(7 * eligible.size / 41))))
    components, component_count = ndimage.label(hot, structure=np.ones((3, 3, 3), dtype=np.uint8))
    per_slice = np.zeros((component_count, n_slices), dtype=np.int32)
    for z in range(n_slices):
        per_slice[:, z] = np.bincount(components[:, :, z].ravel(), minlength=component_count + 1)[1:]
    best = None
    for start in range(int(eligible[0]), int(eligible[-1]) - window + 2):
        component_counts = per_slice[:, start:start + window].sum(axis=1)
        candidate = (
            int(component_counts.max()) if component_count else 0,
            int(hot[:, :, start:start + window].sum()),
            -start,
        )
        if best is None or candidate > best[0]:
            best = (candidate, start)
    if best is None:
        raise ValueError("could not select a contiguous window")
    start = int(best[1])
    return {
        "threshold_99_85": threshold,
        "n_slices": n_slices,
        "nonempty_slices": int(nonempty.size),
        "discarded_leading_nonempty_slices": discarded,
        "eligible_nonempty_slices": int(eligible.size),
        "selected_slices": window,
        "start_zero_based": start,
        "end_zero_based": start + window - 1,
        "start_one_based": start + 1,
        "end_one_based": start + window,
        "hot_voxels_total": int(hot.sum()),
    }


def preprocess_one(payload: tuple[int, str, str, int, tuple[int, int, int]]) -> tuple[int, str, int, np.ndarray, dict[str, int | float]]:
    index, path, uid, label, target = payload
    image = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    selection = choose_window(data)
    start = int(selection["start_zero_based"])
    end = int(selection["end_zero_based"]) + 1
    volume = np.where(np.isfinite(data[:, :, start:end]), np.maximum(data[:, :, start:end], 0), 0).astype(np.float32)
    volume = np.transpose(volume, (2, 1, 0))
    positive = volume[volume > 0]
    if positive.size:
        low, high = np.percentile(positive, [1, 99])
        if high <= low:
            high = low + 1.0
        volume = np.clip((volume - low) / (high - low), 0.0, 1.0)
    resized = ndimage.zoom(volume, tuple(target[i] / volume.shape[i] for i in range(3)), order=1)
    resized = resized.astype(np.float32, copy=False)
    return index, uid, label, resized, selection


class CBAM3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool3d(x, 1)
        maximum = F.adaptive_max_pool3d(x, 1)
        channel_attention = torch.sigmoid(self.channel(avg) + self.channel(maximum))
        x = x * channel_attention
        spatial_input = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.spatial(spatial_input))


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int] = (1, 1, 1)) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.cbam = CBAM3D(out_channels)
        self.downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(out + identity, inplace=True)
        return self.cbam(out)


class ResNet18CBAM3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = self._make_layer(64, 2, (1, 1, 1))
        self.layer2 = self._make_layer(128, 2, (2, 2, 2))
        self.layer3 = self._make_layer(256, 2, (2, 2, 2))
        self.layer4 = self._make_layer(512, 2, (2, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 2)

    def _make_layer(self, channels: int, blocks: int, stride: tuple[int, int, int]) -> nn.Sequential:
        layers = [BasicBlock3D(self.in_channels, channels, stride)]
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_true.extend(y.numpy().tolist())
            y_score.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, pred)),
        "auroc": float(roc_auc_score(truth, score)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    return metrics, pred, score


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    root = args.root.resolve()
    output = root / "results/ants_registered_consecutive_resnet18_cbam"
    output.mkdir(parents=True, exist_ok=True)
    registered = root / "results/ants_registration_mni"
    target = (args.target_depth, args.target_height, args.target_width)

    with (root / "data/external/drive_inputs/train_labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = {row["uid"]: int(float(row["is_pathologic"])) for row in csv.DictReader(handle)}
    paths = sorted(registered.glob("*.nii.gz"))
    payloads = [(i, str(path), path.name.removesuffix(".nii.gz"), labels[path.name.removesuffix(".nii.gz")], target) for i, path in enumerate(paths) if path.name.removesuffix(".nii.gz") in labels]

    X = np.empty((len(payloads), *target), dtype=np.float32)
    uids = [""] * len(payloads)
    y = np.empty(len(payloads), dtype=np.int64)
    selections: list[dict[str, object] | None] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(preprocess_one, payload) for payload in payloads]
        for count, future in enumerate(as_completed(futures), start=1):
            index, uid, label, volume, selection = future.result()
            X[index] = volume
            uids[index] = uid
            y[index] = label
            selections[index] = selection
            if count % 100 == 0 or count == len(futures):
                print(f"preprocessed={count}/{len(futures)}", flush=True)
    np.save(output / "input_windows.npy", X)
    with (output / "slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["uid", "is_pathologic"] + list(selections[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for uid, label, selection in zip(uids, y, selections):
            writer.writerow({"uid": uid, "is_pathologic": int(label), **selection})

    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(all_idx, test_size=0.30, stratify=y, random_state=args.random_state)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.20, stratify=y[train_idx], random_state=args.random_state)
    split = np.full(len(y), "train", dtype=object)
    split[val_idx] = "validation"
    split[test_idx] = "test"
    with (output / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "is_pathologic", "split"])
        writer.writerows([[uid, int(label), part] for uid, label, part in zip(uids, y, split)])

    tensor_x = torch.from_numpy(X).unsqueeze(1)
    tensor_y = torch.from_numpy(y)
    train_loader = DataLoader(TensorDataset(tensor_x[train_idx], tensor_y[train_idx]), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(TensorDataset(tensor_x[val_idx], tensor_y[val_idx]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(TensorDataset(tensor_x[test_idx], tensor_y[test_idx]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18CBAM3D().to(device)
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
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * batch_x.shape[0]
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auroc"])
        row = {"epoch": epoch, "train_loss": running_loss / len(train_idx), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val_metrics.items()}}
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
    torch.save({"model_state_dict": model.state_dict(), "target_shape": target, "args": vars(args), "best_val_auroc": best_val_auc}, output / "best_model.pt")

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    test_rows = []
    for position, index in enumerate(test_idx):
        test_rows.append({"uid": uids[index], "actual": int(y[index]), "predicted": int(predictions[position]), "score_patient": float(scores[position])})
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    metrics = {
        "model": "3D ResNet-18 + CBAM",
        "regularization": "L2 via AdamW weight_decay",
        "weight_decay": args.weight_decay,
        "device": str(device),
        "n_scans": len(y), "n_train": len(train_idx), "n_validation": len(val_idx), "n_test": len(test_idx),
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
        "# Registered consecutive-window 3D ResNet-18 with CBAM\n\n"
        "- Input: ANTs-registered MNI volumes\n"
        "- Slice selection: top 0.15% hotspot scoring, leading 4% of nonempty slices discarded, contiguous window approximately 7/41 of remaining slices\n"
        f"- Resampled input: `{target[0]} x {target[1]} x {target[2]}` (depth x height x width)\n"
        "- Model: one-channel 3D ResNet-18 with CBAM attention after every residual block\n"
        f"- Regularization: L2 weight decay via AdamW, `{args.weight_decay}`; dropout 0.3\n"
        f"- Split: stratified 70/30 test split, with 20% of the training portion used for validation; random_state={args.random_state}\n"
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
