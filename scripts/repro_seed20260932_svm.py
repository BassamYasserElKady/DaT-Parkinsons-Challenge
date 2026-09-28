#!/usr/bin/env python3
"""Regenerate 22 tabular features and rerun the Gaussian RBF SVM with a new seed."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from morphometric_histogram_top_bins_cv5 import retain_top_bins, segment, shape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/repro_seed20260932/svm"
REGISTERED = ROOT / "results/ants_registration_mni"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
WINDOWS = ROOT / "results/morphometric_unequal_two_band_all1362_cv5/window_sources.csv"
BRAIN = ROOT / "assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz"
SEED = 20260932


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = probability >= 0.5
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, prediction)),
        "incorrect": int(np.sum(prediction != y)),
    }


def extract_features(rows: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """Recompute the exact 12 shape + 10 regional-intensity features."""
    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    output = []
    for count, uid in enumerate(rows.uid.astype(str), 1):
        volume = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        start, end = int(windows.loc[uid, "start"]), int(windows.loc[uid, "end"])
        raw = volume[:, :, start : end + 1].max(axis=2)
        brain_projection = brain[:, :, start : end + 1].any(axis=2)
        contour = segment(retain_top_bins(raw, brain_projection, 96), brain_projection)
        grid = np.indices(contour.shape)
        sides = (contour & (grid[0] >= 45), contour & (grid[0] < 45))
        shape_features = shape(sides[0]) + shape(sides[1])
        positive = raw[brain_projection & (raw > 0)]
        normalized = raw / max(float(np.median(positive)), 1e-8)
        regional_means = []
        for side in sides:
            axis_zero = np.where(side)[0]
            if len(axis_zero) < 3:
                regional_means.extend((0.0, 0.0))
                continue
            split = axis_zero.min() + 0.70 * (axis_zero.max() - axis_zero.min())
            upper = side & (grid[1] <= split)
            lower = side & (grid[1] > split)
            regional_means.extend((
                float(normalized[upper].mean()) if upper.any() else 0.0,
                float(normalized[lower].mean()) if lower.any() else 0.0,
            ))
        differences = [
            regional_means[i] - regional_means[j]
            for i in range(4) for j in range(i + 1, 4)
        ]
        output.append(np.r_[shape_features, regional_means, differences])
        if count % 100 == 0 or count == len(rows):
            print(f"features={count}/{len(rows)}", flush=True)
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (len(rows), 22) or not np.all(np.isfinite(features)):
        raise RuntimeError(f"Invalid feature matrix: {features.shape}")
    return features


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ROWS, dtype={"uid": str}).rename(columns={"is_pathologic": "actual"})
    rows = rows.sort_values("uid").reset_index(drop=True)
    windows = pd.read_csv(WINDOWS, dtype={"uid": str}).set_index("uid")
    if set(rows.uid) != set(windows.index):
        raise RuntimeError("Window and cohort UID sets differ.")
    features = extract_features(rows, windows)
    np.save(OUT / "features.npy", features)
    y = rows.actual.to_numpy(dtype=np.int64)
    oof = np.full(len(y), np.nan)
    fold_ids = np.empty(len(y), dtype=np.int64)
    fold_results = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (train, validation) in enumerate(splitter.split(features, y), 1):
        fold_ids[validation] = fold
        model = make_pipeline(
            StandardScaler(),
            SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=SEED + fold),
        )
        model.fit(features[train], y[train])
        oof[validation] = model.predict_proba(features[validation])[:, 1]
        result = {"fold": fold, **metrics(y[validation], oof[validation])}
        fold_results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "seed": SEED,
        "n_scans": len(y),
        "features": 22,
        "feature_definition": "12 bilateral contour shape + 4 regional intensity means + 6 pairwise differences",
        "model": "StandardScaler + SVC(C=1, kernel=rbf, gamma=scale, probability=True)",
        "oof": metrics(y, oof),
        "folds": fold_results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output = rows[["uid", "actual"]].copy()
    output["fold"] = fold_ids
    output["svm"] = oof
    output.to_csv(OUT / "oof_predictions.csv", index=False)
    print(json.dumps(summary["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
