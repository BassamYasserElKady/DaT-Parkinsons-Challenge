#!/usr/bin/env python3
"""Five-fold ResNet-only ablation for post-registration Gaussian smoothing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import cv5_real8_post_registration_gaussian as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_resnet_gaussian_ablation"
ROWS_PATH = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
REGISTERED = ROOT / "results/ants_registration_mni"
SEED = 20260931


def run_sigma(sigma, rows, y, device):
    sigma_name = str(sigma).replace(".", "p")
    sigma_out = OUT / f"sigma_{sigma_name}"
    sigma_out.mkdir(parents=True, exist_ok=True)
    base.OUT = sigma_out
    base.REGISTERED = REGISTERED
    base.SIGMA = sigma
    base.INPUT_PATH = sigma_out / "native_windows.npy"
    native = base.build_inputs(rows)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y))
    oof = np.full(len(y), np.nan)
    folds = []
    for fold, (train_index, outer_index) in enumerate(splits, 1):
        x, means, stds = base.standardize_and_resize(native, train_index)
        print(f"sigma={sigma} fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        model = base.train_independent(base.ResNet18CBAM3D(), x, y, train_index, 8, 16, 1e-3, device, SEED + fold, f"sigma_{sigma_name}_resnet_f{fold}")
        oof[outer_index] = base.predict(model, x, outer_index, device)
        del model, x
        torch.cuda.empty_cache()
        record = {"fold": fold, "train": len(train_index), "outer": len(outer_index), "resnet": base.metrics(y[outer_index], oof[outer_index])}
        folds.append(record)
        (sigma_out / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(sigma_out / "oof_progress.npz", resnet=oof)
        print(json.dumps(record), flush=True)
    summary = {"sigma_voxels": sigma, "model": "3D ResNet-18 + CBAM", "n_scans": len(y), "input": "8 adaptive registered slices, post-registration Gaussian smoothing", "training_augmentation": "horizontal flip only", "oof": base.metrics(y, oof), "folds": folds}
    (sigma_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (sigma_out / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "fold", "resnet_probability"]); writer.writeheader()
        for i, row in enumerate(rows): writer.writerow({"uid": row["uid"], "actual": int(y[i]), "fold": next(f + 1 for f, (_, outer) in enumerate(splits) if i in outer), "resnet_probability": float(oof[i])})
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(ROWS_PATH.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.seed_all(SEED)
    summaries = [run_sigma(sigma, rows, y, device) for sigma in (0.5, 1.0)]
    (OUT / "summary.json").write_text(json.dumps({"device": str(device), "results": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__": main()
