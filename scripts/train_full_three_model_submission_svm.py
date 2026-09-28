#!/usr/bin/env python3
"""Fit the full-data morphometric RBF-SVM and final CNN/SVM meta blend."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "results/repro_seed20260932/svm/features.npy"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OOF = ROOT / "results/real_resnet_densenet_svm_logloss_ensemble/best_oof_predictions.csv"
OUT = ROOT / "results/full_three_model_submission_svm"
SEED = 20260931
EPS = 1e-6


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "incorrect": int(np.sum((p >= 0.5) != y)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS, dtype={"uid": str}).sort_values("uid").reset_index(drop=True)
    x = np.load(FEATURES).astype(np.float64, copy=False)
    y = labels.is_pathologic.to_numpy(dtype=np.int64)
    if x.shape != (len(y), 22):
        raise RuntimeError(f"Unexpected feature shape: {x.shape}")
    model = make_pipeline(
        StandardScaler(),
        SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=SEED),
    )
    model.fit(x, y)
    joblib.dump(model, OUT / "morphometric_intensity_rbf_svm.joblib", compress=3)

    oof = pd.read_csv(OOF, dtype={"uid": str}).sort_values("uid").reset_index(drop=True)
    if not np.array_equal(labels.uid.to_numpy(), oof.uid.to_numpy()):
        raise RuntimeError("OOF and feature UID order differ")
    if not np.array_equal(y, oof.actual.to_numpy(dtype=np.int64)):
        raise RuntimeError("OOF and feature labels differ")
    cnn = 0.5 * (oof.resnet.to_numpy() + oof.densenet.to_numpy())
    svm = oof.svm.to_numpy()
    result = minimize_scalar(
        lambda w: log_loss(y, np.clip(w * cnn + (1 - w) * svm, EPS, 1 - EPS)),
        bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12},
    )
    cnn_weight = float(result.x)
    raw_blend = np.clip(cnn_weight * cnn + (1 - cnn_weight) * svm, EPS, 1 - EPS)
    logits = (np.log(raw_blend) - np.log1p(-raw_blend)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(logits, y)
    calibrated = calibrator.predict_proba(logits)[:, 1]
    metadata = {
        "training_scan_count": int(len(y)),
        "svm": "StandardScaler + SVC(C=1, RBF, gamma=scale, probability=True)",
        "features": "12 contour shape + 4 regional intensity means + 6 pairwise differences",
        "svm_localizer": "adaptive 5-slice, sigma=0.5, percentile=95; middle-50 fallback",
        "cnn_weight": cnn_weight,
        "resnet_weight": cnn_weight / 2,
        "densenet_weight": cnn_weight / 2,
        "svm_weight": 1 - cnn_weight,
        "platt_slope": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "oof_raw_blend": metric(y, raw_blend),
        "oof_calibrated_blend_apparent_fit": metric(y, calibrated),
        "selection_evidence": "pipeline selected by five-fold cross-fitted OOF evaluation; final coefficients fit on all OOF predictions",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
