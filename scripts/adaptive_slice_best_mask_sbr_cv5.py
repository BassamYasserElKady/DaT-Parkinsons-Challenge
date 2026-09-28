#!/usr/bin/env python3
"""Compare full-volume and exact adaptive-five-slice SBR features in nested CV."""
from pathlib import Path
import json
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import grid_mask_shape_sbr_cv5 as grid
import nested_repositioned_mask0_sbr_cv5 as base


ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results/ants_registration_mni"
LABELS = ROOT / "results/cv5_submission_flip_clip/oof_predictions.csv"
MASK_DIR = ROOT / "assets/atlases/fsl/masks"
OUT = ROOT / "results/adaptive_slice_best_mask_sbr_cv5"
LOCALIZER_DIR = ROOT / "submission_ensemble_flip_clip_005_995/src"
SEED = 20260931

sys.path.insert(0, str(LOCALIZER_DIR))
from adaptive_localizer import adaptive_localization  # noqa: E402


def load_or_compute_windows(labels):
    path = OUT / "adaptive_windows.csv"
    if path.exists():
        cached = pd.read_csv(path, dtype={"uid": str})
        if set(cached.uid) == set(labels.uid.astype(str)):
            print(f"loaded adaptive windows: {len(cached)}", flush=True)
            return cached.set_index("uid")

    rows = []
    for i, uid in enumerate(labels.uid.astype(str), 1):
        volume = np.asarray(nib.load(str(REG / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        best, _ = adaptive_localization(volume)
        rows.append({
            "uid": uid,
            "start": int(best["start"]),
            "end": int(best["end"]),
            "detected": int(best["detected"]),
            "localizer_score": float(best["score"]),
        })
        if i % 100 == 0 or i == len(labels):
            print(f"localized={i}/{len(labels)}", flush=True)
    windows = pd.DataFrame(rows)
    windows.to_csv(path, index=False)
    return windows.set_index("uid")


def make_model(fold):
    return XGBClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED + fold,
        n_jobs=4,
    )


def extract_features(labels, brain, occ, caudate, putamen, windows, adaptive, reference_center=None):
    rows = []
    empty_caudate = 0
    empty_putamen = 0
    for uid in labels.uid.astype(str):
        volume = grid.scaled(uid, brain)
        if adaptive:
            start = int(windows.loc[uid, "start"])
            end = int(windows.loc[uid, "end"])
            scan_center = (start + end) / 2.0
            z_shift = int(np.rint(scan_center - reference_center))
            ca = base.translate(caudate, np.array([0, 0, z_shift], dtype=int))
            pu = base.translate(putamen, np.array([0, 0, z_shift], dtype=int))
        else:
            start, end = 0, volume.shape[2] - 1
            z_shift = 0
            ca, pu = caudate, putamen

        if not ca.any():
            empty_caudate += 1
        if not pu.any():
            empty_putamen += 1
        if not ca.any() or not pu.any():
            rows.append([np.nan, np.nan, np.nan, ca.sum(), pu.sum(), start, end, z_shift])
            continue

        occ_mean = float(volume[occ].mean())
        ca_mean = float(volume[ca].mean())
        pu_mean = float(volume[pu].mean())
        st_mean = float(volume[ca | pu].mean())
        rows.append([
            (ca_mean - occ_mean) / occ_mean,
            (pu_mean - occ_mean) / occ_mean,
            (st_mean - occ_mean) / occ_mean,
            int(ca.sum()), int(pu.sum()), start, end, z_shift,
        ])
    columns = ["caudate_sbr", "putamen_sbr", "striatum_sbr",
               "caudate_voxels", "putamen_voxels", "start", "end", "z_shift"]
    return pd.DataFrame(rows, columns=columns), empty_caudate, empty_putamen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS, dtype={"uid": str}).sort_values("uid").reset_index(drop=True)
    y = labels.actual.to_numpy(dtype=int)
    windows = load_or_compute_windows(labels)
    brain = np.asarray(nib.load(str(MASK_DIR / "MNI152_T1_2mm_brain_mask.nii.gz")).dataobj) > 0
    occ = np.asarray(nib.load(str(MASK_DIR / "occipital_cortex.nii.gz")).dataobj) > 0
    names = ["left_caudate", "right_caudate", "left_putamen", "right_putamen"]
    original = {name: np.asarray(nib.load(str(MASK_DIR / f"{name}.nii.gz")).dataobj) > 0 for name in names}
    xcoord = np.indices(brain.shape)[0]

    methods = ("full_3d", "adaptive_z_aligned_roi")
    oof = {method: np.full(len(labels), np.nan) for method in methods}
    diagnostic_rows = []
    feature_rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(labels, y), 1):
        healthy_uids = labels.iloc[train_idx].loc[labels.iloc[train_idx].actual == 0, "uid"].astype(str)
        total = np.zeros(brain.shape, dtype=np.float64)
        for uid in healthy_uids:
            total += grid.scaled(uid, brain)
        template = (total / len(healthy_uids)).astype(np.float32)
        reference_window, _ = adaptive_localization(template)
        reference_center = (int(reference_window["start"]) + int(reference_window["end"])) / 2.0
        base_ca, base_pu, shifts = base.reposition(template, original)
        caudate = (grid.circularize(base_ca & (xcoord >= 45), 0.9) |
                   grid.circularize(base_ca & (xcoord < 45), 0.9))
        putamen = (grid.shorten(base_pu & (xcoord >= 45), 0.6) |
                   grid.shorten(base_pu & (xcoord < 45), 0.6)) & ~caudate

        for method in methods:
            feats, empty_ca, empty_pu = extract_features(
                labels, brain, occ, caudate, putamen, windows,
                method == "adaptive_z_aligned_roi", reference_center
            )
            feature_cols = ["caudate_sbr", "putamen_sbr", "striatum_sbr"]
            valid_features = np.isfinite(feats[feature_cols].to_numpy()).all(axis=1)
            if not valid_features.all():
                raise RuntimeError(
                    f"EMPTY_ADAPTIVE_MASK fold={fold} method={method} "
                    f"empty_caudate={empty_ca} empty_putamen={empty_pu}"
                )
            model = make_model(fold)
            model.fit(feats.loc[train_idx, feature_cols], y[train_idx])
            oof[method][valid_idx] = model.predict_proba(feats.loc[valid_idx, feature_cols])[:, 1]
            diagnostic_rows.append({
                "fold": fold,
                "method": method,
                "healthy_reference_scans": len(healthy_uids),
                "empty_caudate": empty_ca,
                "empty_putamen": empty_pu,
                "median_caudate_voxels": float(feats.caudate_voxels.median()),
                "median_putamen_voxels": float(feats.putamen_voxels.median()),
                "reference_window_start": int(reference_window["start"]),
                "reference_window_end": int(reference_window["end"]),
                "median_z_shift": float(feats.z_shift.median()),
                "min_z_shift": int(feats.z_shift.min()),
                "max_z_shift": int(feats.z_shift.max()),
                "mask_shifts": json.dumps(shifts),
            })
            fold_pred = oof[method][valid_idx]
            print(
                f"fold={fold} method={method} auroc={roc_auc_score(y[valid_idx], fold_pred):.6f} "
                f"logloss={log_loss(y[valid_idx], fold_pred, labels=[0, 1]):.6f} "
                f"accuracy={accuracy_score(y[valid_idx], fold_pred >= 0.5):.6f}",
                flush=True,
            )
            selected = feats.iloc[valid_idx].copy()
            selected.insert(0, "uid", labels.iloc[valid_idx].uid.to_numpy())
            selected.insert(1, "fold", fold)
            selected.insert(2, "method", method)
            feature_rows.append(selected)

    metric_rows = []
    prediction_frames = []
    for method, probability in oof.items():
        prediction = (probability >= 0.5).astype(int)
        metric_rows.append({
            "method": method,
            "n": len(y),
            "auroc": roc_auc_score(y, probability),
            "log_loss": log_loss(y, probability, labels=[0, 1]),
            "accuracy": accuracy_score(y, prediction),
            "incorrect": int((prediction != y).sum()),
        })
        prediction_frames.append(pd.DataFrame({
            "uid": labels.uid, "actual": y, "method": method,
            "oof_probability": probability, "oof_prediction": prediction,
        }))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(OUT / "fold_diagnostics.csv", index=False)
    pd.concat(feature_rows, ignore_index=True).to_csv(OUT / "fold_features.csv", index=False)
    print("\nFINAL", flush=True)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
