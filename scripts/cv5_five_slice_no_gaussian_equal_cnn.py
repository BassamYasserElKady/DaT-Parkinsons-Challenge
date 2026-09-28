#!/usr/bin/env python3
"""Five-fold equal ResNet/DenseNet ensemble on the original five-slice inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv5_original_flip_vs_dml import (
    DenseNet121_3D,
    ResNet18CBAM3D,
    predict,
    seed_all,
    train_independent,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_five_slice_no_gaussian_equal_cnn"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
INPUTS = ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy"
SEED = 20260931


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p >= 0.5
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "incorrect": int(np.sum(pred != y)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    x = np.load(INPUTS).astype(np.float32, copy=False)
    if x.shape[0] != len(y) or x.shape[1:] != (16, 64, 64):
        raise RuntimeError(f"Expected resized five-slice inputs for {len(y)} scans, got {x.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y))
    resnet_oof = np.full(len(y), np.nan, dtype=np.float64)
    densenet_oof = np.full(len(y), np.nan, dtype=np.float64)
    ensemble_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.empty(len(y), dtype=np.int64)
    for fold, (_, test) in enumerate(splits, 1):
        fold_ids[test] = fold
    progress_path = OUT / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed = {int(row["fold"]) for row in progress}
    if (OUT / "oof_progress.npz").exists():
        saved = np.load(OUT / "oof_progress.npz")
        resnet_oof = saved["resnet"]
        densenet_oof = saved["densenet"]
        ensemble_oof = saved["equal_cnn"]
    folds = progress

    print(f"device={device} input={INPUTS} shape={x.shape} gaussian_sigma=0 completed={sorted(completed)}", flush=True)
    fold_order = [1, 4, 2, 3, 5]
    for fold in fold_order:
        if fold in completed:
            continue
        train, test = splits[fold - 1]
        print(f"fold={fold}/5 train={len(train)} test={len(test)}", flush=True)
        seed_all(SEED + fold)
        resnet = train_independent(
            ResNet18CBAM3D(), x, y, train, 8, 16, 1e-3, device,
            SEED + fold, f"five_no_gaussian_resnet_f{fold}",
        )
        resnet_oof[test] = predict(resnet, x, test, device)
        del resnet
        if device.type == "cuda":
            torch.cuda.empty_cache()

        densenet = train_independent(
            DenseNet121_3D(), x, y, train, 2, 14, 3e-4, device,
            SEED + fold, f"five_no_gaussian_densenet_f{fold}",
        )
        densenet_oof[test] = predict(densenet, x, test, device)
        del densenet
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ensemble_oof[test] = 0.5 * resnet_oof[test] + 0.5 * densenet_oof[test]
        record = {
            "fold": fold,
            "resnet": metrics(y[test], resnet_oof[test]),
            "densenet": metrics(y[test], densenet_oof[test]),
            "equal_cnn": metrics(y[test], ensemble_oof[test]),
        }
        folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "oof_progress.npz", resnet=resnet_oof, densenet=densenet_oof,
                 equal_cnn=ensemble_oof, fold=fold_ids)
        print(json.dumps(record), flush=True)

    summary = {
        "n_scans": len(y),
        "input": "adaptive five real axial slices; no Gaussian smoothing",
        "gaussian_sigma_voxels": 0.0,
        "training_augmentation": "random horizontal flip only",
        "resnet_epochs": 16,
        "densenet_epochs": 14,
        "ensemble_weighting": "0.5 ResNet + 0.5 DenseNet",
        "resnet_oof": metrics(y, resnet_oof),
        "densenet_oof": metrics(y, densenet_oof),
        "equal_ensemble_oof": metrics(y, ensemble_oof),
        "folds": folds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "fold", "resnet", "densenet", "equal_cnn"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"uid": row["uid"], "actual": int(y[index]), "fold": int(fold_ids[index]),
                             "resnet": float(resnet_oof[index]), "densenet": float(densenet_oof[index]),
                             "equal_cnn": float(ensemble_oof[index])})
    print(json.dumps({"equal_ensemble_oof": summary["equal_ensemble_oof"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
