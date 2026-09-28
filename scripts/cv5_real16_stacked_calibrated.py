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
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
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
OUT = ROOT / "results/cv5_real16_stacked_calibrated"
REGISTERED = ROOT / "results/ants_registration_mni"
ROWS_PATH = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
INPUT_PATH = OUT / "real16_per_volume_zscore.npy"
SEED = 20260931
DEPTH = 16
EPS = 1e-6


def preprocess_one(payload):
    index, uid, label = payload
    localizer_dir = str(ROOT / "original_submission_files")
    if localizer_dir not in sys.path:
        sys.path.insert(0, localizer_dir)
    from adaptive_localizer import adaptive_localization

    volume = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).get_fdata(), dtype=np.float64)
    best, _ = adaptive_localization(volume)
    center = (int(best["start"]) + int(best["end"])) // 2
    start = min(max(center - DEPTH // 2, 0), volume.shape[2] - DEPTH)
    end = start + DEPTH
    valid = np.isfinite(volume) & (volume > 0)
    positive = volume[valid]
    if positive.size == 0 or float(positive.std()) < 1e-8:
        raise RuntimeError(f"invalid positive voxels for {uid}")
    normalized = np.zeros(volume.shape, dtype=np.float32)
    normalized[valid] = ((volume[valid] - float(positive.mean())) / float(positive.std())).astype(np.float32)
    window = np.stack([normalized[:, :, z].T for z in range(start, end)], axis=0)
    resized = ndimage.zoom(window, (1.0, 64 / window.shape[1], 64 / window.shape[2]), order=1).astype(np.float32)
    if resized.shape != (DEPTH, 64, 64) or not np.all(np.isfinite(resized)):
        raise RuntimeError(f"invalid input for {uid}: {resized.shape}")
    return index, uid, label, resized, start, end - 1, center


def build_inputs(rows, jobs=8):
    OUT.mkdir(parents=True, exist_ok=True)
    if INPUT_PATH.exists():
        x = np.load(INPUT_PATH).astype(np.float32, copy=False)
        print(f"reusing_inputs={INPUT_PATH} shape={x.shape}", flush=True)
        return x
    x = np.empty((len(rows), DEPTH, 64, 64), dtype=np.float32)
    audit = [None] * len(rows)
    payloads = [(i, row["uid"], int(row["is_pathologic"])) for i, row in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(preprocess_one, payload) for payload in payloads]
        for count, future in enumerate(as_completed(futures), 1):
            index, uid, label, volume, start, end, center = future.result()
            x[index] = volume
            audit[index] = {"uid": uid, "is_pathologic": label, "start": start, "end": end, "center": center}
            if count % 100 == 0 or count == len(rows):
                print(f"preprocessed={count}/{len(rows)}", flush=True)
    np.save(INPUT_PATH, x)
    with (OUT / "slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    return x


def fold_standardize(x, train_index):
    means = x[train_index].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    stds = x[train_index].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    if np.any(stds < 1e-8):
        raise RuntimeError("invalid fold standard deviation")
    standardized = (x - means[None, :, None, None]) / stds[None, :, None, None]
    return standardized.astype(np.float32, copy=False), means, stds


def bounded_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def metric(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, np.asarray(p) >= 0.5)),
        "incorrect": int(np.sum((np.asarray(p) >= 0.5) != y)),
    }


def stack_features(resnet_p, dense_p):
    return np.column_stack((bounded_logit(resnet_p), bounded_logit(dense_p)))


def beta_features(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.column_stack((np.log(p), -np.log1p(-p)))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader(ROWS_PATH.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    x = build_inputs(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(x, y))
    fold_id = np.empty(len(y), dtype=np.int64)
    resnet_oof = np.full(len(y), np.nan)
    dense_oof = np.full(len(y), np.nan)
    fold_records = []

    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_id[outer_index] = fold
        fold_x, means, stds = fold_standardize(x, train_index)
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        resnet = train_independent(
            ResNet18CBAM3D(), fold_x, y, train_index, 8, 16, 1e-3,
            device, SEED + fold, f"real16_resnet_f{fold}",
        )
        resnet_oof[outer_index] = predict(resnet, fold_x, outer_index, device)
        del resnet
        torch.cuda.empty_cache()
        dense = train_independent(
            DenseNet121_3D(), fold_x, y, train_index, 2, 14, 3e-4,
            device, SEED + fold, f"real16_dense_f{fold}",
        )
        dense_oof[outer_index] = predict(dense, fold_x, outer_index, device)
        del dense, fold_x
        torch.cuda.empty_cache()
        record = {
            "fold": fold,
            "train": int(len(train_index)),
            "outer": int(len(outer_index)),
            "channel_means": means.tolist(),
            "channel_stds": stds.tolist(),
            "resnet": metric(y[outer_index], resnet_oof[outer_index]),
            "densenet": metric(y[outer_index], dense_oof[outer_index]),
        }
        fold_records.append(record)
        (OUT / "base_progress.json").write_text(json.dumps(fold_records, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "base_oof_progress.npz", resnet=resnet_oof, densenet=dense_oof, fold=fold_id)
        print(json.dumps(record), flush=True)

    selected_oof = np.full(len(y), np.nan)
    raw_stack_oof = np.full(len(y), np.nan)
    temperature_oof = np.full(len(y), np.nan)
    beta_oof = np.full(len(y), np.nan)
    meta_records = []
    for fold in range(1, 6):
        test_index = np.flatnonzero(fold_id == fold)
        meta_index = np.flatnonzero(fold_id != fold)
        stack = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=SEED + fold)
        stack.fit(stack_features(resnet_oof[meta_index], dense_oof[meta_index]), y[meta_index])
        meta_raw = stack.predict_proba(stack_features(resnet_oof[meta_index], dense_oof[meta_index]))[:, 1]
        test_raw = stack.predict_proba(stack_features(resnet_oof[test_index], dense_oof[test_index]))[:, 1]
        raw_stack_oof[test_index] = test_raw

        optimum = minimize_scalar(
            lambda log_t: log_loss(y[meta_index], sigmoid(bounded_logit(meta_raw) / np.exp(log_t)), labels=[0, 1]),
            bounds=(-3.0, 3.0), method="bounded",
        )
        temperature = float(np.exp(optimum.x))
        meta_temp = sigmoid(bounded_logit(meta_raw) / temperature)
        test_temp = sigmoid(bounded_logit(test_raw) / temperature)
        temperature_oof[test_index] = test_temp

        beta = LogisticRegression(C=10.0, solver="lbfgs", max_iter=3000, random_state=SEED + 100 + fold)
        beta.fit(beta_features(meta_raw), y[meta_index])
        meta_beta = beta.predict_proba(beta_features(meta_raw))[:, 1]
        test_beta = beta.predict_proba(beta_features(test_raw))[:, 1]
        beta_oof[test_index] = test_beta

        candidates = {
            "raw": (log_loss(y[meta_index], meta_raw, labels=[0, 1]), test_raw),
            "temperature": (log_loss(y[meta_index], meta_temp, labels=[0, 1]), test_temp),
            "beta": (log_loss(y[meta_index], meta_beta, labels=[0, 1]), test_beta),
        }
        method = min(candidates, key=lambda name: candidates[name][0])
        selected_oof[test_index] = candidates[method][1]
        meta_records.append({
            "fold": fold,
            "method": method,
            "temperature": temperature,
            "stack_coefficients": stack.coef_[0].tolist(),
            "stack_intercept": float(stack.intercept_[0]),
            "meta_log_loss": {name: float(value[0]) for name, value in candidates.items()},
        })

    clipped = np.clip(selected_oof, 0.005, 0.995)
    summary = {
        "n_scans": len(y),
        "input": "16 real consecutive axial slices centered on adaptive five-slice localization",
        "fold_specific_standardization": True,
        "training_augmentation": "horizontal flip only",
        "device": str(device),
        "resnet_oof": metric(y, resnet_oof),
        "densenet_oof": metric(y, dense_oof),
        "raw_logistic_stack_oof": metric(y, raw_stack_oof),
        "temperature_stack_oof": metric(y, temperature_oof),
        "beta_stack_oof": metric(y, beta_oof),
        "crossfit_selected_calibration_oof": metric(y, selected_oof),
        "crossfit_selected_calibration_clipped_005_995_oof": metric(y, clipped),
        "base_folds": fold_records,
        "meta_folds": meta_records,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["uid", "actual", "fold", "resnet", "densenet", "raw_stack", "temperature", "beta", "selected", "selected_clipped"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "uid": row["uid"], "actual": int(y[i]), "fold": int(fold_id[i]),
                "resnet": resnet_oof[i], "densenet": dense_oof[i], "raw_stack": raw_stack_oof[i],
                "temperature": temperature_oof[i], "beta": beta_oof[i],
                "selected": selected_oof[i], "selected_clipped": clipped[i],
            })
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("oof")}, indent=2), flush=True)
    print(f"output={OUT}", flush=True)


if __name__ == "__main__":
    main()
