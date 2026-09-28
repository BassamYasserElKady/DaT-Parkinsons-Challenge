#!/usr/bin/env python3
"""Leakage-controlled five-fold SVM comparison with explicit occipital SBRs."""

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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_svm_occipital_sbr"
REGISTERED = ROOT / "results/ants_registration_mni"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
BRAIN = ROOT / "assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz"
MASK_DIR = ROOT / "assets/atlases/fsl/masks"
SEED = 20260931


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p >= 0.5
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "incorrect": int(np.sum(pred != y)),
    }


def make_features(rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # Import the exact adaptive localizer and 22-feature implementation used by
    # the current five-slice SVM. The feature code itself is label-free.
    import sys
    sys.path.insert(0, str(ROOT / "submission_ensemble_sigma075_8slice_svm"))
    from src.svm_features import adaptive_window, extract_features

    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    mask_names = ["left_caudate", "right_caudate", "left_putamen", "right_putamen"]
    masks = [np.asarray(nib.load(str(MASK_DIR / f"{name}.nii.gz")).dataobj) > 0
             for name in mask_names]
    occipital = np.asarray(nib.load(str(MASK_DIR / "occipital_cortex.nii.gz")).dataobj) > 0

    features22: list[np.ndarray] = []
    extra: list[list[float]] = []
    for count, uid in enumerate(rows.uid.astype(str), 1):
        volume = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        base, info = extract_features(volume, brain)
        start, end = int(info["start"]), int(info["end"])
        raw = volume[:, :, start:end + 1].max(axis=2)
        brain2 = brain[:, :, start:end + 1].any(axis=2)
        roi_masks = [m[:, :, start:end + 1].any(axis=2) for m in masks]
        occ = occipital[:, :, start:end + 1].any(axis=2)
        positive = raw[brain2 & np.isfinite(raw) & (raw > 0)]
        occ_values = raw[occ & np.isfinite(raw) & (raw > 0)]
        occ_mean = float(occ_values.mean()) if occ_values.size else 0.0
        occ_std = float(occ_values.std()) if occ_values.size else 0.0
        denom = max(occ_mean, 1e-8)
        roi_means = [
            float(raw[m & np.isfinite(raw)].mean()) if np.any(m & np.isfinite(raw)) else 0.0
            for m in roi_masks
        ]
        sbr = [(value - occ_mean) / denom for value in roi_means]
        caudate_mean = float(np.mean(sbr[:2]))
        putamen_mean = float(np.mean(sbr[2:]))
        caudate_asym = (sbr[0] - sbr[1]) / max(abs(sbr[0]) + abs(sbr[1]), 1e-8)
        putamen_asym = (sbr[2] - sbr[3]) / max(abs(sbr[2]) + abs(sbr[3]), 1e-8)
        features22.append(base)
        extra.append([occ_mean, occ_std, *sbr, caudate_mean, putamen_mean,
                      caudate_asym, putamen_asym])
        if count % 100 == 0 or count == len(rows):
            print(f"features={count}/{len(rows)}", flush=True)

    x22 = np.asarray(features22, dtype=np.float64)
    xextra = np.asarray(extra, dtype=np.float64)
    x31 = np.column_stack([x22, xextra])
    if x22.shape != (len(rows), 22) or x31.shape != (len(rows), 32):
        raise RuntimeError(f"Unexpected feature shapes: {x22.shape}, {x31.shape}")
    if not np.all(np.isfinite(x31)):
        raise RuntimeError("Non-finite SVM features")
    names = [f"existing_{i:02d}" for i in range(22)] + [
        "occipital_mean", "occipital_std", "left_caudate_sbr", "right_caudate_sbr",
        "left_putamen_sbr", "right_putamen_sbr", "mean_caudate_sbr",
        "mean_putamen_sbr", "caudate_asymmetry_sbr", "putamen_asymmetry_sbr",
    ]
    return np.column_stack([x22, x31]), names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ROWS, dtype={"uid": str}).rename(columns={"is_pathologic": "actual"})
    rows = rows.sort_values("uid").reset_index(drop=True)
    combined, names = make_features(rows)
    x22, x32 = combined[:, :22], combined[:, 22:]
    np.savez_compressed(OUT / "features.npz", existing22=x22, occipital_sbr32=x32)
    y = rows.actual.to_numpy(dtype=np.int64)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = {"existing22": np.full(len(y), np.nan), "occipital_sbr32": np.full(len(y), np.nan)}
    fold_results = []
    for fold, (train, validation) in enumerate(splitter.split(x22, y), 1):
        result = {"fold": fold}
        for name, x in [("existing22", x22), ("occipital_sbr32", x32)]:
            model = make_pipeline(StandardScaler(), SVC(
                C=1, kernel="rbf", gamma="scale", probability=True,
                random_state=SEED + fold,
            ))
            model.fit(x[train], y[train])
            oof[name][validation] = model.predict_proba(x[validation])[:, 1]
            result[name] = metric(y[validation], oof[name][validation])
        fold_results.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "seed": SEED,
        "n_scans": len(y),
        "existing_features": 22,
        "new_features": 32,
        "new_feature_names": names[22:],
        "model": "StandardScaler + SVC(C=1, kernel=rbf, gamma=scale, probability=True)",
        "pooled": {name: metric(y, p) for name, p in oof.items()},
        "folds": fold_results,
        "note": "Masks and adaptive windows are label-free; scaler is fit within each training fold.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output = rows[["uid", "actual"]].copy()
    output["fold"] = np.nan
    output["existing22"] = oof["existing22"]
    output["occipital_sbr32"] = oof["occipital_sbr32"]
    output.to_csv(OUT / "oof_predictions.csv", index=False)
    print(json.dumps(summary["pooled"], indent=2), flush=True)


if __name__ == "__main__":
    main()
