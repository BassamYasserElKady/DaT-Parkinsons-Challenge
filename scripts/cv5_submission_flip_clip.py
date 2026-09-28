#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv5_original_flip_vs_dml import (
    DenseNet121_3D,
    ResNet18CBAM3D,
    calibrated,
    predict,
    train_independent,
    seed_all,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cv5_submission_flip_clip"
SEED = 20260931
LOW = 0.005
HIGH = 0.995


def metrics(y, p):
    pred = p >= 0.5
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "correct": int(np.sum(pred == y)),
        "incorrect": int(np.sum(pred != y)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader((ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open()))
    y = np.asarray([int(r["is_pathologic"]) for r in rows], dtype=np.int64)
    x = np.load(ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy").astype(np.float32, copy=False)
    if len(x) != len(y):
        raise RuntimeError(f"input/label count mismatch: {len(x)} vs {len(y)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    folds = []

    for fold, (outer_train, outer) in enumerate(skf.split(x, y), 1):
        train, val = train_test_split(outer_train, test_size=0.2, stratify=y[outer_train], random_state=SEED + fold)
        print(f"fold={fold}/5 train={len(train)} val={len(val)} outer={len(outer)}", flush=True)
        r = train_independent(ResNet18CBAM3D(), x, y, train, 8, 16, 1e-3, device, SEED + fold, f"resnet18_f{fold}")
        rv, ro = predict(r, x, val, device), predict(r, x, outer, device)
        del r
        torch.cuda.empty_cache()
        d = train_independent(DenseNet121_3D(), x, y, train, 2, 14, 3e-4, device, SEED + fold, f"densenet121_f{fold}")
        dv, do = predict(d, x, val, device), predict(d, x, outer, device)
        del d
        torch.cuda.empty_cache()

        weight = float(minimize_scalar(lambda w: log_loss(y[val], w * rv + (1 - w) * dv, labels=[0, 1]), bounds=(0, 1), method="bounded").x)
        val_raw = weight * rv + (1 - weight) * dv
        outer_raw = weight * ro + (1 - weight) * do
        outer_calibrated = calibrated(y[val], val_raw, outer_raw)
        outer_clipped = np.clip(outer_calibrated, LOW, HIGH)
        oof[outer] = outer_clipped
        fold_metrics = metrics(y[outer], outer_clipped)
        fold_record = {"fold": fold, "resnet_weight": weight, "densenet_weight": 1 - weight, **fold_metrics}
        folds.append(fold_record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n")
        print(json.dumps(fold_record), flush=True)

    pred = oof >= 0.5
    wrong = [
        {"uid": rows[i]["uid"], "actual": int(y[i]), "prediction": int(pred[i]), "probability": float(oof[i])}
        for i in range(len(y)) if pred[i] != y[i]
    ]
    summary = {
        "n_scans": len(y),
        "folds": 5,
        "inner_validation_fraction": 0.2,
        "clip": [LOW, HIGH],
        "oof": metrics(y, oof),
        "folds_detail": folds,
        "misclassified_count": len(wrong),
        "misclassified_scans": wrong,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "oof_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "actual", "prediction", "probability", "fold"])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"uid": row["uid"], "actual": int(y[i]), "prediction": int(pred[i]), "probability": float(oof[i]), "fold": next(j + 1 for j, (_, outer) in enumerate(skf.split(x, y)) if i in outer)})
    print(json.dumps({"oof": summary["oof"], "misclassified_count": len(wrong)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
