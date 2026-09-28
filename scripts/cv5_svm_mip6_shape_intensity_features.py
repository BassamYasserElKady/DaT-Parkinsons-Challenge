#!/usr/bin/env python3
"""Leakage-controlled five-fold RBF-SVM test of guided 3D shape/intensity features."""
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
BASE = ROOT / "results" / "cv5_svm_occipital_sbr" / "features.npz"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
SHAPES = ROOT / "results" / "adaptive8_mip6_guided_3d_quarter_shapes" / "labels"
MAPS = ROOT / "results" / "adaptive8_mip6_guided_sliding_intensity_pyramid" / "maps"
OUT = ROOT / "results" / "cv5_svm_mip6_shape_intensity_features"
SEED = 20260931
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy": float(accuracy_score(y, p >= .5)), "incorrect": int(np.sum((p >= .5) != y))}


def distribution(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0] * (2 + len(QUANTILES))
    return [float(values.mean()), float(values.std()), *[float(np.quantile(values, q)) for q in QUANTILES]]


def extra_features(uid: str) -> tuple[np.ndarray, list[str]]:
    label = np.asarray(nib.load(str(SHAPES / f"{uid}_adaptive8_3d_quarters.nii.gz")).dataobj, dtype=np.uint8)
    left_counts = np.asarray([np.sum(label == code) for code in range(1, 5)], dtype=float)
    right_counts = np.asarray([np.sum(label == code) for code in range(5, 9)], dtype=float)
    left_total, right_total = float(left_counts.sum()), float(right_counts.sum())
    shape = np.r_[left_total, right_total, left_counts, right_counts,
                  left_counts / max(left_total, 1.0), right_counts / max(right_total, 1.0)]
    shape_names = ["left_volume", "right_volume"] + [f"left_q{i}_volume" for i in range(1, 5)] + [f"right_q{i}_volume" for i in range(1, 5)] + [f"left_q{i}_fraction" for i in range(1, 5)] + [f"right_q{i}_fraction" for i in range(1, 5)]
    sparse = np.load(MAPS / f"{uid}_adaptive8_sliding_pyramid.npz")
    chunks, names = [shape], shape_names
    # Six maps: three local scales for each side. Values are normalized by the
    # scan's median raw contour intensity to represent relative distribution.
    raw = sparse["intensity_1"].astype(float); positive = raw[raw > 0]
    scale = max(float(np.median(positive)) if len(positive) else 1.0, 1e-6)
    for side_code, side_name in ((0, "left"), (1, "right")):
        selected = sparse["side"] == side_code
        for level, key in enumerate(("intensity_1", "intensity_2", "intensity_3"), 1):
            chunks.append(np.asarray(distribution(sparse[key][selected].astype(float) / scale), dtype=float))
            names += [f"{side_name}_l{level}_{stat}" for stat in ("mean", "std", "q05", "q25", "q50", "q75", "q95")]
    return np.concatenate(chunks), names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(LABELS, dtype={"uid": str}).sort_values("uid").reset_index(drop=True)
    y = rows.is_pathologic.to_numpy(dtype=np.int64)
    existing = np.load(BASE)["existing22"].astype(float)
    if len(existing) != len(rows): raise RuntimeError("existing22 and labels not aligned")
    extras, names = [], None
    for index, uid in enumerate(rows.uid, 1):
        values, local_names = extra_features(uid); extras.append(values); names = local_names
        if index % 100 == 0 or index == len(rows): print(f"features={index}/{len(rows)}", flush=True)
    added = np.asarray(extras, dtype=float)
    combined = np.column_stack((existing, added))
    if not np.all(np.isfinite(combined)): raise RuntimeError("non-finite feature")
    np.savez_compressed(OUT / "features.npz", existing22=existing, added_mip6_shape_intensity=added, combined=combined)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    predictions = {"existing22": np.full(len(y), np.nan), "plus_mip6_shape_intensity": np.full(len(y), np.nan)}
    folds=[]
    for fold, (train, valid) in enumerate(splitter.split(combined, y), 1):
        record={"fold": fold}
        for name, x in (("existing22", existing), ("plus_mip6_shape_intensity", combined)):
            model=make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED+fold))
            model.fit(x[train], y[train]); predictions[name][valid]=model.predict_proba(x[valid])[:,1]
            record[name]=metric(y[valid], predictions[name][valid])
        folds.append(record); print(json.dumps(record), flush=True)
    summary={"evaluation":"five-fold out-of-fold; StandardScaler and SVM fit only on each fold's training scans", "base_features":22, "added_features":int(added.shape[1]), "combined_features":int(combined.shape[1]), "added_feature_names":names, "model":"RBF SVC(C=1, gamma=scale, probability=True)", "pooled":{k:metric(y,v) for k,v in predictions.items()}, "folds":folds}
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    output=rows[["uid","is_pathologic"]].copy(); output["fold"]=0
    for fold, (_,valid) in enumerate(splitter.split(combined,y),1): output.loc[valid,"fold"]=fold
    for name,p in predictions.items(): output[name]=p
    output.to_csv(OUT / "oof_predictions.csv",index=False)
    print(json.dumps(summary["pooled"],indent=2),flush=True)


if __name__ == "__main__": main()
