#!/usr/bin/env python3
"""Train 3D ResNet-18 + CBAM using only top-0.25% binary masks."""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D, evaluate, set_seed


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


def choose_middle_nonempty_mask(data: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0).astype(np.float32, copy=False)
    threshold = float(np.percentile(data, 99.75))
    hot = data >= threshold
    nonempty = np.flatnonzero(np.max(data, axis=(0, 1)) > 0)
    if nonempty.size == 0:
        raise ValueError("scan has no nonempty slices")
    target_count = max(1, min(int(nonempty.size), int(round(7 * nonempty.size / 41))))
    band_start_rank = int(ceil(0.25 * (nonempty.size - 1)))
    band_end_rank = int(np.floor(0.75 * (nonempty.size - 1)))
    band_start_rank = min(band_start_rank, int(nonempty.size - target_count))
    band_end_rank = max(band_end_rank, band_start_rank + target_count - 1)

    labels, component_count = ndimage.label(hot, structure=np.ones((3, 3, 3), dtype=np.uint8))
    per_slice = np.zeros((component_count, data.shape[2]), dtype=np.int32)
    for z in range(data.shape[2]):
        per_slice[:, z] = np.bincount(labels[:, :, z].ravel(), minlength=component_count + 1)[1:]

    best = None
    for start_rank in range(band_start_rank, band_end_rank - target_count + 2):
        selected = nonempty[start_rank:start_rank + target_count]
        counts = per_slice[:, selected].sum(axis=1)
        candidate = (
            int(counts.max()) if component_count else 0,
            int(hot[:, :, selected].sum()),
            -int(selected[0]),
        )
        if best is None or candidate > best[0]:
            best = (candidate, selected.copy(), start_rank)
    if best is None:
        raise ValueError("could not select a middle nonempty-slice window")
    selected = best[1]
    return hot[:, :, selected], {
        "threshold_99_75": threshold,
        "n_slices": int(data.shape[2]),
        "nonempty_slices": int(nonempty.size),
        "middle_band_start_rank_1based": band_start_rank + 1,
        "middle_band_end_rank_1based": band_end_rank + 1,
        "middle_band_start_slice_1based": int(nonempty[band_start_rank]) + 1,
        "middle_band_end_slice_1based": int(nonempty[band_end_rank]) + 1,
        "selected_nonempty_count": int(selected.size),
        "selected_slice_indices_1based": ";".join(str(int(z) + 1) for z in selected),
        "selected_rank_start_1based": int(best[2]) + 1,
        "selected_rank_end_1based": int(best[2] + selected.size),
        "selected_mask_voxels": int(hot[:, :, selected].sum()),
    }


def resize_binary(volume: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    factors = tuple(target[i] / volume.shape[i] for i in range(3))
    resized = ndimage.zoom(volume.astype(np.float32), factors, order=0)
    output = np.zeros(target, dtype=np.float32)
    slices_in = tuple(slice(0, min(resized.shape[i], target[i])) for i in range(3))
    slices_out = tuple(slice(0, min(resized.shape[i], target[i])) for i in range(3))
    output[slices_out] = resized[slices_in]
    return output


def preprocess_one(payload: tuple[int, str, str, int, tuple[int, int, int]]) -> tuple[int, str, int, np.ndarray, dict[str, object]]:
    index, path, uid, label, target = payload
    data = np.asarray(nib.as_closest_canonical(nib.load(path)).get_fdata(), dtype=np.float32)
    mask, selection = choose_middle_nonempty_mask(data)
    volume = np.transpose(mask.astype(np.float32), (2, 1, 0))
    return index, uid, label, resize_binary(volume, target), selection


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    root = args.root.resolve()
    output = root / "results/ants_registered_top025_middle_nonempty_mask_resnet18_cbam"
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
    np.save(output / "binary_mask_windows.npy", X)
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
    pin = torch.cuda.is_available()
    train_loader = DataLoader(TensorDataset(tensor_x[train_idx], tensor_y[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(tensor_x[val_idx], tensor_y[val_idx]), batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(TensorDataset(tensor_x[test_idx], tensor_y[test_idx]), batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18CBAM3D().to(device)
    counts = np.bincount(y[train_idx], minlength=2).astype(float)
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_auc = -np.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * batch_x.shape[0]
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auroc"])
        row = {"epoch": epoch, "train_loss": total_loss / len(train_idx), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} val_auc={val_metrics['auroc']:.4f} val_acc={val_metrics['accuracy']:.4f}", flush=True)
        if val_metrics["auroc"] > best_auc:
            best_auc = val_metrics["auroc"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("early_stopping=true", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, predictions, scores = evaluate(model, test_loader, device)
    torch.save({"model_state_dict": model.state_dict(), "target_shape": target, "args": vars(args), "best_val_auroc": best_auc}, output / "best_model.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "actual", "predicted", "score_patient"])
        writer.writerows([[uids[index], int(y[index]), int(predictions[position]), float(scores[position])] for position, index in enumerate(test_idx)])
    metrics = {
        "model": "3D ResNet-18 + CBAM on binary top-0.25% mask",
        "regularization": "L2 via AdamW weight_decay",
        "weight_decay": args.weight_decay,
        "device": str(device),
        "n_scans": len(y), "n_train": len(train_idx), "n_validation": len(val_idx), "n_test": len(test_idx),
        "patients_total": int(y.sum()), "healthy_total": int((y == 0).sum()),
        "patients_test": int(y[test_idx].sum()), "healthy_test": int((y[test_idx] == 0).sum()),
        "best_validation_auroc": float(best_auc), **test_metrics,
    }
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    (output / "report.md").write_text(
        "# Registered middle nonempty-slice binary-mask ResNet-18 with CBAM\n\n"
        "- Input: only a binary mask of voxels at or above each scan's 99.75th percentile\n"
        "- Slice rule: nonempty slices only; candidate window constrained between the 25th and 75th nonempty-slice ranks; selected count approximately 7/41 of nonempty slices\n"
        f"- Resampled input: `{target[0]} x {target[1]} x {target[2]}` (depth x height x width), nearest-neighbour resizing\n"
        "- Model: one-channel 3D ResNet-18 with CBAM after every residual block\n"
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
