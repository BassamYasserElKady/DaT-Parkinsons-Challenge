#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
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
OUT = ROOT / "results/cv5_real8_interp16"
REGISTERED = ROOT / "results/ants_registration_mni"
ROWS_PATH = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
INPUT_PATH = OUT / "real8_native_per_volume_zscore.npy"
SEED = 20260931
NATIVE_DEPTH = 8
TARGET_SHAPE = (16, 64, 64)


def preprocess_one(payload):
    index, uid, label = payload
    localizer_dir = str(ROOT / "original_submission_files")
    if localizer_dir not in sys.path:
        sys.path.insert(0, localizer_dir)
    from adaptive_localizer import adaptive_localization

    volume = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).get_fdata(), dtype=np.float64)
    best, _ = adaptive_localization(volume)
    center = (int(best["start"]) + int(best["end"])) // 2
    start = min(max(center - NATIVE_DEPTH // 2, 0), volume.shape[2] - NATIVE_DEPTH)
    end = start + NATIVE_DEPTH
    valid = np.isfinite(volume) & (volume > 0)
    positive = volume[valid]
    if positive.size == 0 or float(positive.std()) < 1e-8:
        raise RuntimeError(f"invalid positive voxels for {uid}")
    normalized = np.zeros(volume.shape, dtype=np.float32)
    normalized[valid] = ((volume[valid] - float(positive.mean())) / float(positive.std())).astype(np.float32)
    window = np.stack([normalized[:, :, z].T for z in range(start, end)], axis=0).astype(np.float32)
    if window.shape != (NATIVE_DEPTH, 109, 91) or not np.all(np.isfinite(window)):
        raise RuntimeError(f"invalid native window for {uid}: {window.shape}")
    return index, uid, label, window, start, end - 1, center


def build_inputs(rows, jobs=8):
    OUT.mkdir(parents=True, exist_ok=True)
    if INPUT_PATH.exists():
        x = np.load(INPUT_PATH).astype(np.float32, copy=False)
        print(f"reusing_inputs={INPUT_PATH} shape={x.shape}", flush=True)
        return x
    x = np.empty((len(rows), NATIVE_DEPTH, 109, 91), dtype=np.float32)
    audit = [None] * len(rows)
    payloads = [(i, row["uid"], int(row["is_pathologic"])) for i, row in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(preprocess_one, payload) for payload in payloads]
        for count, future in enumerate(as_completed(futures), 1):
            index, uid, label, window, start, end, center = future.result()
            x[index] = window
            audit[index] = {"uid": uid, "is_pathologic": label, "start": start, "end": end, "center": center}
            if count % 100 == 0 or count == len(rows):
                print(f"preprocessed={count}/{len(rows)}", flush=True)
    np.save(INPUT_PATH, x)
    with (OUT / "slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    return x


def standardize_and_resize(native, train_index):
    means = native[train_index].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    stds = native[train_index].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    if np.any(stds < 1e-8):
        raise RuntimeError("invalid fold standard deviation")
    output = np.empty((len(native), *TARGET_SHAPE), dtype=np.float32)
    zoom = (TARGET_SHAPE[0] / NATIVE_DEPTH, TARGET_SHAPE[1] / 109, TARGET_SHAPE[2] / 91)
    for index in range(len(native)):
        standardized = (native[index] - means[:, None, None]) / stds[:, None, None]
        output[index] = ndimage.zoom(standardized, zoom, order=1).astype(np.float32)
    return output, means, stds


def metrics(y, p):
    prediction = np.asarray(p) >= 0.5
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, prediction)),
        "incorrect": int(np.sum(prediction != y)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader(ROWS_PATH.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = build_inputs(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y))
    fold_id = np.empty(len(y), dtype=np.int64)
    resnet_oof = np.full(len(y), np.nan)
    dense_oof = np.full(len(y), np.nan)
    folds = []

    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_id[outer_index] = fold
        x, means, stds = standardize_and_resize(native, train_index)
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        resnet = train_independent(
            ResNet18CBAM3D(), x, y, train_index, 8, 16, 1e-3,
            device, SEED + fold, f"real8_interp16_resnet_f{fold}",
        )
        resnet_oof[outer_index] = predict(resnet, x, outer_index, device)
        del resnet
        torch.cuda.empty_cache()
        dense = train_independent(
            DenseNet121_3D(), x, y, train_index, 2, 14, 3e-4,
            device, SEED + fold, f"real8_interp16_dense_f{fold}",
        )
        dense_oof[outer_index] = predict(dense, x, outer_index, device)
        del dense, x
        torch.cuda.empty_cache()
        ensemble = 0.5 * resnet_oof[outer_index] + 0.5 * dense_oof[outer_index]
        record = {
            "fold": fold,
            "train": int(len(train_index)),
            "outer": int(len(outer_index)),
            "channel_means": means.tolist(),
            "channel_stds": stds.tolist(),
            "resnet": metrics(y[outer_index], resnet_oof[outer_index]),
            "densenet": metrics(y[outer_index], dense_oof[outer_index]),
            "equal_ensemble": metrics(y[outer_index], ensemble),
        }
        folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "oof_progress.npz", resnet=resnet_oof, densenet=dense_oof, fold=fold_id)
        print(json.dumps(record), flush=True)

    ensemble_oof = 0.5 * resnet_oof + 0.5 * dense_oof
    clipped_oof = np.clip(ensemble_oof, 0.005, 0.995)
    summary = {
        "n_scans": len(y),
        "input": "8 real consecutive slices linearly interpolated to 16",
        "fold_specific_standardization": True,
        "training_augmentation": "horizontal flip only",
        "resnet_oof": metrics(y, resnet_oof),
        "densenet_oof": metrics(y, dense_oof),
        "equal_ensemble_oof": metrics(y, ensemble_oof),
        "equal_ensemble_clipped_005_995_oof": metrics(y, clipped_oof),
        "folds": folds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["uid", "actual", "fold", "resnet", "densenet", "equal_ensemble", "equal_ensemble_clipped"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "uid": row["uid"], "actual": int(y[i]), "fold": int(fold_id[i]),
                "resnet": resnet_oof[i], "densenet": dense_oof[i],
                "equal_ensemble": ensemble_oof[i], "equal_ensemble_clipped": clipped_oof[i],
            })
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("oof")}, indent=2), flush=True)
    print(f"output={OUT}", flush=True)


if __name__ == "__main__":
    main()
