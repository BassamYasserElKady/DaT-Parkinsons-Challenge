#!/usr/bin/env python3
"""Retrain the 3D ResNet/DenseNet ensemble with the downloaded preprocessing.

The ANTs default-Rigid registrations in ``results/ants_registration_mni`` were
created by the same call as the downloaded ``register_to_template.py``. This
script applies the downloaded adaptive five-slice localizer and standardization,
resizes the resulting five-slice volume to the existing models' 16x64x64 input,
trains both models on the established split, selects a soft-voting weight on
validation AUROC, and evaluates the held-out test split once.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resnet-batch-size", type=int, default=8)
    parser.add_argument("--densenet-batch-size", type=int, default=2)
    parser.add_argument("--target-depth", type=int, default=16)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rebuild-inputs", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_one(payload: tuple[int, str, str, int, tuple[int, int, int], list[float], list[float], str]):
    index, path, uid, label, target, means, stds, localizer_dir = payload
    if localizer_dir not in sys.path:
        sys.path.insert(0, localizer_dir)
    from adaptive_localizer import adaptive_localization

    image = nib.load(path)
    volume = np.asarray(image.get_fdata(), dtype=np.float64)
    best, _ = adaptive_localization(volume)
    start, end = int(best["start"]), int(best["end"])
    positive_mask = np.isfinite(volume) & (volume > 0)
    positive = volume[positive_mask]
    if positive.size == 0:
        raise RuntimeError(f"EMPTY_POSITIVE_REGISTERED_VOLUME:{uid}")
    mean, std = float(positive.mean()), float(positive.std())
    if not np.isfinite(std) or std < 1e-8:
        raise RuntimeError(f"INVALID_VOLUME_STANDARDIZATION:{uid}")
    normalized = np.zeros(volume.shape, dtype=np.float32)
    normalized[positive_mask] = ((volume[positive_mask] - mean) / std).astype(np.float32)
    slices = np.stack([normalized[:, :, z].T for z in range(start, end + 1)], axis=0)
    if slices.shape != (5, 109, 91):
        raise RuntimeError(f"UNEXPECTED_ADAPTIVE_SHAPE:{uid}:{slices.shape}")
    standardized = (slices - np.asarray(means, dtype=np.float32)[:, None, None]) / np.asarray(stds, dtype=np.float32)[:, None, None]
    resized = ndimage.zoom(standardized, tuple(target[i] / standardized.shape[i] for i in range(3)), order=1)
    resized = resized.astype(np.float32, copy=False)
    if resized.shape != target or not np.all(np.isfinite(resized)):
        raise RuntimeError(f"INVALID_RESIZED_INPUT:{uid}:{resized.shape}")
    return index, uid, label, resized, start, end, int(best["detected"]), float(best["score"])


def metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    prediction = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, prediction)),
        "auroc": float(roc_auc_score(truth, scores)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def score_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device, non_blocking=True))
            truth.extend(batch_y.numpy().tolist())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(truth, dtype=int), np.asarray(scores, dtype=float)


def train_model(name: str, model: nn.Module, loaders: dict[str, DataLoader], y_train: np.ndarray, device: torch.device, output: Path, epochs: int, patience: int, lr: float) -> dict[str, object]:
    model = model.to(device)
    class_counts = np.bincount(y_train, minlength=2).astype(float)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * class_counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_auc, stale, history = None, -np.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(batch_y)
        val_truth, val_scores = score_model(model, loaders["validation"], device)
        val = metrics(val_truth, val_scores)
        scheduler.step(val["auroc"])
        row = {"epoch": epoch, "train_loss": total_loss / len(y_train), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"model={name} epoch={epoch}/{epochs} loss={row['train_loss']:.4f} val_auc={val['auroc']:.4f} val_acc={val['accuracy']:.4f}", flush=True)
        if val["auroc"] > best_auc:
            best_auc, best_state, stale = float(val["auroc"]), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print(f"model={name} early_stopping=true", flush=True)
                break
    model.load_state_dict(best_state)
    val_truth, val_scores = score_model(model, loaders["validation"], device)
    test_truth, test_scores = score_model(model, loaders["test"], device)
    torch.save({"model_state_dict": model.state_dict(), "best_validation_auroc": best_auc, "input_shape": (16, 64, 64)}, output / f"{name}_best_model.pt")
    with (output / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    return {"name": name, "best_validation_auroc": best_auc, "validation_truth": val_truth, "validation_scores": val_scores, "test_truth": test_truth, "test_scores": test_scores}


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    root = args.root.resolve()
    output = root / "results/downloaded_rigid_adaptive_3d_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "results/ants_registration_mni"
    original = root / "original_submission_files"
    target = (args.target_depth, args.target_height, args.target_width)
    input_path = output / "input_windows.npy"
    split_source = root / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
    split_rows = list(csv.DictReader(split_source.open(newline="", encoding="utf-8")))
    config = json.loads((original / "standardization.json").read_text(encoding="utf-8"))
    means, stds = config["channel_means"], config["channel_stds"]

    if args.rebuild_inputs or not input_path.exists():
        X = np.empty((len(split_rows), *target), dtype=np.float32)
        audit = [None] * len(split_rows)
        payloads = [(i, str(source / f"{row['uid']}.nii.gz"), row["uid"], int(row["is_pathologic"]), target, means, stds, str(original)) for i, row in enumerate(split_rows)]
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(preprocess_one, payload) for payload in payloads]
            for count, future in enumerate(as_completed(futures), start=1):
                index, uid, label, volume, start, end, detected, score = future.result()
                X[index] = volume
                audit[index] = {"uid": uid, "is_pathologic": label, "start": start, "end": end, "detected": detected, "localizer_score": score}
                if count % 100 == 0 or count == len(futures):
                    print(f"preprocessed={count}/{len(futures)}", flush=True)
        np.save(input_path, X)
        with (output / "adaptive_slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader(); writer.writerows(audit)
    else:
        X = np.load(input_path).astype(np.float32, copy=False)
        print(f"reusing_inputs={input_path} shape={X.shape}", flush=True)

    y = np.asarray([int(row["is_pathologic"]) for row in split_rows], dtype=np.int64)
    parts = np.asarray([row["split"] for row in split_rows])
    indices = {part: np.flatnonzero(parts == part) for part in ("train", "validation", "test")}
    tensor_x, tensor_y = torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)
    pin = torch.cuda.is_available()
    def loaders(batch_size: int) -> dict[str, DataLoader]:
        return {part: DataLoader(TensorDataset(tensor_x[idx], tensor_y[idx]), batch_size=batch_size, shuffle=part == "train", num_workers=0, pin_memory=pin) for part, idx in indices.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = train_model("resnet18_cbam_3d", ResNet18CBAM3D(), loaders(args.resnet_batch_size), y[indices["train"]], device, output, args.epochs, args.patience, 1e-3)
    del tensor_x
    torch.cuda.empty_cache()
    tensor_x = torch.from_numpy(X).unsqueeze(1)
    dense = train_model("densenet121_3d", DenseNet121_3D(), loaders(args.densenet_batch_size), y[indices["train"]], device, output, args.epochs, args.patience, 3e-4)

    if not np.array_equal(resnet["validation_truth"], dense["validation_truth"]) or not np.array_equal(resnet["test_truth"], dense["test_truth"]):
        raise RuntimeError("MODEL_LABEL_ORDER_MISMATCH")
    sweep = []
    for weight in np.linspace(0.0, 1.0, 1001):
        scores = weight * resnet["validation_scores"] + (1.0 - weight) * dense["validation_scores"]
        sweep.append({"resnet_weight": float(weight), "densenet_weight": float(1.0 - weight), "validation_auroc": float(roc_auc_score(resnet["validation_truth"], scores)), "validation_accuracy": float(accuracy_score(resnet["validation_truth"], scores >= 0.5))})
    best = max(sweep, key=lambda row: (row["validation_auroc"], row["validation_accuracy"], -abs(row["resnet_weight"] - 0.5)))
    weight = best["resnet_weight"]
    test_scores = weight * resnet["test_scores"] + (1.0 - weight) * dense["test_scores"]
    result_rows = []
    for label, score in (("3D ResNet-18 + CBAM", resnet["test_scores"]), ("3D DenseNet-121", dense["test_scores"]), ("Validation-selected weighted ensemble", test_scores)):
        result_rows.append({"model": label, "resnet_weight": weight if "ensemble" in label else "", "densenet_weight": 1.0 - weight if "ensemble" in label else "", "n_test": len(resnet["test_truth"]), **metrics(resnet["test_truth"], score)})
    with (output / "weight_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0])); writer.writeheader(); writer.writerows(sweep)
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    test_idx = indices["test"]
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["uid", "actual", "resnet_score", "densenet_score", "ensemble_score", "ensemble_predicted"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for pos, index in enumerate(test_idx):
            writer.writerow({"uid": split_rows[index]["uid"], "actual": int(y[index]), "resnet_score": float(resnet["test_scores"][pos]), "densenet_score": float(dense["test_scores"][pos]), "ensemble_score": float(test_scores[pos]), "ensemble_predicted": int(test_scores[pos] >= 0.5)})
    report = [
        "# Downloaded rigid-registration adaptive-slice 3D ensemble\n",
        "- Registration: existing ANTsPy default `Rigid` outputs, matching the downloaded registration call.",
        "- Localization/normalization: downloaded label-free five-slice adaptive localizer and supplied channel standardization.",
        "- Model compatibility resize: 5x109x91 to 16x64x64; model architectures unchanged.",
        "- Split: unchanged established train/validation/test split (762/191/409).",
        f"- Validation-selected weights: ResNet-18 `{weight:.3f}`, DenseNet-121 `{1.0-weight:.3f}`.",
        f"- Validation ensemble AUROC: `{best['validation_auroc']:.4f}`.\n",
        "## Held-out test\n",
    ]
    report.extend(f"- **{row['model']}**: accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}, AUROC={row['auroc']:.4f}" for row in result_rows)
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"selected_weights=resnet={weight:.3f},densenet={1.0-weight:.3f}", flush=True)
    for row in result_rows:
        print(f"{row['model']}: accuracy={row['accuracy']:.4f} auroc={row['auroc']:.4f}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
