#!/usr/bin/env python3
"""Train the unchanged downloaded five-slice ResNet-50 CBAM architecture."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def load_downloaded_model_builder(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("downloaded_resnet50_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load model module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model


def preprocess_one(payload):
    index, path, uid, label, start, end, means, stds = payload
    volume = np.asarray(nib.load(path).get_fdata(), dtype=np.float64)
    valid = np.isfinite(volume) & (volume > 0)
    positive = volume[valid]
    if positive.size == 0:
        raise RuntimeError(f"EMPTY_POSITIVE_REGISTERED_VOLUME:{uid}")
    mean, std = float(positive.mean()), float(positive.std())
    if not np.isfinite(std) or std < 1e-8:
        raise RuntimeError(f"INVALID_VOLUME_STANDARDIZATION:{uid}")
    normalized = np.zeros(volume.shape, dtype=np.float32)
    normalized[valid] = ((volume[valid] - mean) / std).astype(np.float32)
    slices = np.stack([normalized[:, :, z].T for z in range(start, end + 1)], axis=0)
    if slices.shape != (5, 109, 91):
        raise RuntimeError(f"UNEXPECTED_ADAPTIVE_SHAPE:{uid}:{slices.shape}")
    standardized = (slices - np.asarray(means, dtype=np.float32)[:, None, None]) / np.asarray(stds, dtype=np.float32)[:, None, None]
    if not np.all(np.isfinite(standardized)):
        raise RuntimeError(f"NONFINITE_STANDARDIZED_INPUT:{uid}")
    return index, uid, label, standardized.astype(np.float32, copy=False)


def evaluate(model, loader, device):
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for x, y in loader:
            truth.extend(y.numpy().tolist())
            scores.extend(torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1].cpu().numpy().tolist())
    truth, scores = np.asarray(truth, dtype=int), np.asarray(scores, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, predictions)),
        "auroc": float(roc_auc_score(truth, scores)),
        "log_loss": float(log_loss(truth, scores, labels=[0, 1])),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }, predictions, scores


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    root = args.root.resolve()
    originals = root / "original_submission_files"
    output = root / "results/downloaded_resnet50_cbam"
    output.mkdir(parents=True, exist_ok=True)
    split_rows = list(csv.DictReader((root / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open(newline="", encoding="utf-8")))
    selections = {row["uid"]: row for row in csv.DictReader((root / "results/downloaded_rigid_adaptive_3d_ensemble/adaptive_slice_selection.csv").open(newline="", encoding="utf-8"))}
    standardization = json.loads((originals / "standardization.json").read_text(encoding="utf-8"))
    input_path = output / "input_five_slices.npy"
    if args.rebuild_inputs or not input_path.exists():
        X = np.empty((len(split_rows), 5, 109, 91), dtype=np.float32)
        payloads = []
        for index, row in enumerate(split_rows):
            selected = selections[row["uid"]]
            payloads.append((index, str(root / "results/ants_registration_mni" / f"{row['uid']}.nii.gz"), row["uid"], int(row["is_pathologic"]), int(selected["start"]), int(selected["end"]), standardization["channel_means"], standardization["channel_stds"]))
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(preprocess_one, payload) for payload in payloads]
            for count, future in enumerate(as_completed(futures), start=1):
                index, _, _, image = future.result()
                X[index] = image
                if count % 100 == 0 or count == len(futures):
                    print(f"preprocessed={count}/{len(futures)}", flush=True)
        np.save(input_path, X)
    else:
        X = np.load(input_path).astype(np.float32, copy=False)
        print(f"reusing_inputs={input_path} shape={X.shape}", flush=True)

    y = np.asarray([int(row["is_pathologic"]) for row in split_rows], dtype=np.int64)
    parts = np.asarray([row["split"] for row in split_rows])
    indices = {part: np.flatnonzero(parts == part) for part in ("train", "validation", "test")}
    tensor_x, tensor_y = torch.from_numpy(X), torch.from_numpy(y)
    pin = torch.cuda.is_available()
    loaders = {part: DataLoader(TensorDataset(tensor_x[idx], tensor_y[idx]), batch_size=args.batch_size, shuffle=part == "train", num_workers=0, pin_memory=pin) for part, idx in indices.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_model = load_downloaded_model_builder(originals / "model.py")
    model = build_model().to(device)
    class_counts = np.bincount(y[indices["train"]], minlength=2).astype(float)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * class_counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_auc, stale, history = None, -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loaders["train"]:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(batch_y)
        val_metrics, _, _ = evaluate(model, loaders["validation"], device)
        scheduler.step(val_metrics["auroc"])
        row = {"epoch": epoch, "train_loss": total_loss / len(indices["train"]), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} val_auc={val_metrics['auroc']:.4f} val_acc={val_metrics['accuracy']:.4f} val_log_loss={val_metrics['log_loss']:.4f}", flush=True)
        if val_metrics["auroc"] > best_auc:
            best_auc, best_state, stale = float(val_metrics["auroc"]), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                print("early_stopping=true", flush=True)
                break
    model.load_state_dict(best_state)
    test_metrics, predictions, scores = evaluate(model, loaders["test"], device)
    torch.save({"model_state_dict": model.state_dict(), "best_validation_auroc": best_auc, "input_shape": (5, 109, 91), "args": vars(args)}, output / "best_model.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "predicted", "score_patient"]); writer.writeheader()
        for position, index in enumerate(indices["test"]):
            writer.writerow({"uid": split_rows[index]["uid"], "actual": int(y[index]), "predicted": int(predictions[position]), "score_patient": float(scores[position])})
    summary = {"model": "Downloaded 2D ResNet-50 CBAM", "device": str(device), "n_scans": len(y), "n_train": len(indices["train"]), "n_validation": len(indices["validation"]), "n_test": len(indices["test"]), "best_validation_auroc": best_auc, **test_metrics}
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    (output / "report.md").write_text(
        "# Downloaded five-slice ResNet-50 CBAM\n\n"
        "- Architecture: unchanged downloaded `model.py`; random initialization.\n"
        "- Input: downloaded ANTs Rigid registration, adaptive five-slice localization, per-volume nonzero z-score, and frozen channel standardization.\n"
        "- Split: same established train/validation/test split (762/191/409).\n"
        f"- Best checkpoint selected by validation AUROC; maximum epochs={args.epochs}, patience={args.patience}.\n\n"
        "## Held-out test\n\n" + "\n".join(f"- **{key}:** {value:.4f}" if isinstance(value, float) else f"- **{key}:** {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(f"test_metrics={summary}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
