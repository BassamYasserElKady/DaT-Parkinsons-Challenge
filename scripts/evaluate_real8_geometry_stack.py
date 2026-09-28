#!/usr/bin/env python3
"""Fold-safe fusion of the real-8-slice ensemble with four geometry features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "results/cv5_real8_interp16/oof_predictions.csv"
GEOMETRY_PATH = ROOT / "results/top025_full_geometry/geometry_features.csv"
OUT = ROOT / "results/cv5_real8_geometry_stack"
FEATURES = ["mirror_hausdorff95_mm", "mirror_assd_mm", "mirror_dice", "second_component_voxels"]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "incorrect": int(np.sum((p >= 0.5) != y)),
    }


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main():
    oof = pd.read_csv(OOF_PATH)
    geo = pd.read_csv(GEOMETRY_PATH)[["uid", *FEATURES]]
    frame = oof.merge(geo, on="uid", how="left", validate="one_to_one")
    if frame[FEATURES].isna().any().any():
        raise ValueError("Geometry features contain missing values")
    y = frame.actual.to_numpy(dtype=int)
    base = frame.equal_ensemble_clipped.to_numpy(dtype=float)
    # Base scores remain OOF; the stacker is cross-fitted using the existing five folds.
    x = np.column_stack([logit(frame.resnet), logit(frame.densenet), frame[FEATURES].to_numpy(dtype=float)])
    fold_ids = frame.fold.to_numpy(dtype=int)
    stacked = np.full(len(frame), np.nan)
    fold_rows = []
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        scaler = StandardScaler().fit(x[train, 2:])
        xtr = np.column_stack([x[train, :2], scaler.transform(x[train, 2:])])
        xte = np.column_stack([x[test, :2], scaler.transform(x[test, 2:])])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=20260931)
        model.fit(xtr, y[train])
        stacked[test] = model.predict_proba(xte)[:, 1]
        fold_rows.append({
            "fold": int(fold),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "baseline": metrics(y[test], base[test]),
            "geometry_stack": metrics(y[test], stacked[test]),
        })
    summary = {
        "n_scans": int(len(y)),
        "features": FEATURES,
        "method": "Cross-fitted logistic stacker using OOF ResNet/DenseNet scores plus four geometry features; StandardScaler fitted within each fold; probabilities clipped to [0.005, 0.995] for metrics.",
        "baseline_real8_equal_ensemble_clipped": metrics(y, base),
        "geometry_stack_oof": metrics(y, stacked),
        "delta_geometry_minus_baseline": {
            "auroc": float(roc_auc_score(y, stacked) - roc_auc_score(y, base)),
            "log_loss": float(log_loss(y, stacked, labels=[0, 1]) - log_loss(y, base, labels=[0, 1])),
            "accuracy": float(accuracy_score(y, stacked >= 0.5) - accuracy_score(y, base >= 0.5)),
            "incorrect": int(np.sum((stacked >= 0.5) != y) - np.sum((base >= 0.5) != y)),
        },
        "folds": fold_rows,
        "feature_only_note": "The four features were selected from the prior OOF error analysis, so this result is exploratory and should be confirmed with a pre-specified fresh split or repeated CV.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = frame[["uid", "actual", "fold", "resnet", "densenet", "equal_ensemble_clipped", *FEATURES]].copy()
    output["geometry_stack_probability"] = stacked
    output["geometry_stack_prediction"] = (stacked >= 0.5).astype(int)
    output.to_csv(OUT / "oof_predictions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
