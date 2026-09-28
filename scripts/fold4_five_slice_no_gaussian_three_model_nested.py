#!/usr/bin/env python3
"""Leakage-controlled fold-1 rerun for the five-slice three-model ensemble."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv5_original_flip_vs_dml import (
    DenseNet121_3D,
    ResNet18CBAM3D,
    predict,
    seed_all,
    train_independent,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/fold4_five_slice_no_gaussian_three_model_nested"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
INPUTS = ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy"
REGISTERED = ROOT / "results/ants_registration_mni"
BRAIN = ROOT / "submission_ensemble_sigma075_8slice_svm/assets/brain_mask.nii.gz"
PACKAGE = ROOT / "submission_ensemble_sigma075_8slice_svm"
SEED = 20260931
EPS = 1e-6


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p, EPS, 1 - EPS)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "incorrect": int(np.sum((p >= 0.5) != y))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    x = np.load(INPUTS).astype(np.float32, copy=False)
    if x.shape != (len(y), 16, 64, 64):
        raise RuntimeError(f"Unexpected five-slice input shape: {x.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_train, outer_test = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y))[3]
    inner_train, inner_val = train_test_split(
        outer_train, test_size=0.2, stratify=y[outer_train], random_state=SEED + 1)
    print(f"device={device} outer_train={len(outer_train)} inner_train={len(inner_train)} "
          f"inner_val={len(inner_val)} outer_test={len(outer_test)}", flush=True)

    seed_all(SEED + 1)
    resnet = train_independent(ResNet18CBAM3D(), x, y, inner_train, 8, 16, 1e-3,
                               device, SEED + 1, "nested_f4_resnet")
    res_val = predict(resnet, x, inner_val, device)
    res_test = predict(resnet, x, outer_test, device)
    del resnet
    if device.type == "cuda": torch.cuda.empty_cache()

    dense = train_independent(DenseNet121_3D(), x, y, inner_train, 2, 14, 3e-4,
                              device, SEED + 1, "nested_f4_densenet")
    dense_val = predict(dense, x, inner_val, device)
    dense_test = predict(dense, x, outer_test, device)
    del dense
    if device.type == "cuda": torch.cuda.empty_cache()

    import sys
    sys.path.insert(0, str(PACKAGE))
    from src.svm_features import extract_features
    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    features = []
    for count, row in enumerate(rows, 1):
        volume = np.asarray(nib.load(str(REGISTERED / f"{row['uid']}.nii.gz")).dataobj, dtype=np.float64)
        features.append(extract_features(volume, brain)[0])
        if count % 200 == 0 or count == len(rows): print(f"svm_features={count}/{len(rows)}", flush=True)
    features = np.asarray(features, dtype=np.float64)
    svm = make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale",
                                              probability=True, random_state=SEED + 1))
    svm.fit(features[inner_train], y[inner_train])
    svm_val = svm.predict_proba(features[inner_val])[:, 1]
    svm_test = svm.predict_proba(features[outer_test])[:, 1]

    cnn_val = 0.5 * (res_val + dense_val)
    cnn_test = 0.5 * (res_test + dense_test)
    result = minimize_scalar(
        lambda w: log_loss(y[inner_val], np.clip(w * cnn_val + (1 - w) * svm_val, EPS, 1 - EPS)),
        bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
    cnn_weight = float(result.x)
    raw_val = np.clip(cnn_weight * cnn_val + (1 - cnn_weight) * svm_val, EPS, 1 - EPS)
    raw_test = np.clip(cnn_weight * cnn_test + (1 - cnn_weight) * svm_test, EPS, 1 - EPS)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(
        (np.log(raw_val) - np.log1p(-raw_val)).reshape(-1, 1), y[inner_val])
    calibrated_test = calibrator.predict_proba(
        (np.log(raw_test) - np.log1p(-raw_test)).reshape(-1, 1))[:, 1]

    test_uids = np.asarray([rows[i]["uid"] for i in outer_test])
    np.savez(OUT / "predictions.npz", uid=test_uids, actual=y[outer_test], resnet=res_test, densenet=dense_test, svm=svm_test, equal_cnn=cnn_test, calibrated=calibrated_test)

    summary = {
        "input": "adaptive five slices, no Gaussian smoothing",
        "outer_fold": 4, "n_scans": len(y), "inner_train": len(inner_train),
        "inner_validation": len(inner_val), "outer_test": len(outer_test),
        "resnet": metric(y[outer_test], res_test),
        "densenet": metric(y[outer_test], dense_test),
        "svm": metric(y[outer_test], svm_test),
        "equal_cnn": metric(y[outer_test], cnn_test),
        "calibrated_three_model": metric(y[outer_test], calibrated_test),
        "cnn_weight": cnn_weight, "svm_weight": 1 - cnn_weight,
        "platt_slope": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
