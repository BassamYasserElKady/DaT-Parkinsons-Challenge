#!/usr/bin/env python3
"""Resume the independent-seed CNN reproducibility run from completed OOF folds."""

from __future__ import annotations

import csv
import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import cv5_real8_post_registration_gaussian as base
from cv5_original_flip_vs_dml import DenseNet121_3D, ResNet18CBAM3D, predict, seed_all, train_independent
from cv5_real8_interp16 import metrics, standardize_and_resize


ROOT = base.ROOT
OUT = ROOT / "results/repro_seed20260932/cnn"
INPUT = OUT / "real8_native_post_registration_gaussian.npy"
ROWS = base.ROWS_PATH
SEED = 20260932


def main() -> None:
    rows = list(csv.DictReader(ROWS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = np.load(INPUT).astype(np.float32, copy=False)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y))
    saved = np.load(OUT / "oof_progress.npz")
    resnet_oof = saved["resnet"].astype(np.float64)
    dense_oof = saved["densenet"].astype(np.float64)
    fold_id = np.empty(len(y), dtype=np.int64)
    for fold, (_, outer) in enumerate(splits, 1):
        fold_id[outer] = fold
    folds = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
    completed = {int(record["fold"]) for record in folds}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} completed={sorted(completed)}", flush=True)

    for fold, (train_index, outer_index) in enumerate(splits, 1):
        if fold in completed:
            print(f"fold={fold}/5 already_complete", flush=True)
            continue
        seed_all(SEED + fold)
        x, _, _ = standardize_and_resize(native, train_index)
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        resnet = train_independent(
            ResNet18CBAM3D(), x, y, train_index, 8, 16, 1e-3,
            device, SEED + fold, f"repro_resnet_f{fold}",
        )
        resnet_oof[outer_index] = predict(resnet, x, outer_index, device)
        del resnet
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dense = train_independent(
            DenseNet121_3D(), x, y, train_index, 2, 14, 3e-4,
            device, SEED + fold, f"repro_densenet_f{fold}",
        )
        dense_oof[outer_index] = predict(dense, x, outer_index, device)
        del dense, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ensemble = 0.5 * (resnet_oof[outer_index] + dense_oof[outer_index])
        record = {
            "fold": fold,
            "train": int(len(train_index)),
            "outer": int(len(outer_index)),
            "resnet": metrics(y[outer_index], resnet_oof[outer_index]),
            "densenet": metrics(y[outer_index], dense_oof[outer_index]),
            "equal_ensemble": metrics(y[outer_index], ensemble),
        }
        folds.append(record)
        folds.sort(key=lambda item: item["fold"])
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "oof_progress.npz", resnet=resnet_oof, densenet=dense_oof, fold=fold_id)
        print(json.dumps(record), flush=True)

    if not np.all(np.isfinite(resnet_oof)) or not np.all(np.isfinite(dense_oof)):
        raise RuntimeError("OOF arrays remain incomplete after resume.")
    ensemble = 0.5 * (resnet_oof + dense_oof)
    summary = {
        "seed": SEED,
        "n_scans": len(y),
        "input": "rebuilt 8-slice post-registration Gaussian sigma=0.75 inputs",
        "resnet_epochs": 16,
        "densenet_epochs": 14,
        "resnet_oof": metrics(y, resnet_oof),
        "densenet_oof": metrics(y, dense_oof),
        "equal_ensemble_oof": metrics(y, ensemble),
        "folds": folds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["uid", "actual", "fold", "resnet", "densenet", "equal_ensemble"]
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "uid": row["uid"], "actual": int(y[index]), "fold": int(fold_id[index]),
                "resnet": float(resnet_oof[index]), "densenet": float(dense_oof[index]),
                "equal_ensemble": float(ensemble[index]),
            })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
