#!/usr/bin/env python3
"""Extract 2D morphometrics from MIPs of the selected smoothed 8-slice windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OOF = ROOT / "results/cv5_real8_post_registration_gaussian/oof_predictions.csv"
OUT = ROOT / "results/mip_top025_morphometric_features"
FEATURES = ["right_eccentricity", "left_eccentricity", "left_circularity", "right_circularity", "right_area_mm2", "left_area_mm2", "right_minor_axis_length_mm"]


def perimeter(mask: np.ndarray, sx: float, sy: float) -> float:
    left = mask & ~np.roll(mask, 1, axis=0); left[0, :] = mask[0, :]
    right = mask & ~np.roll(mask, -1, axis=0); right[-1, :] = mask[-1, :]
    up = mask & ~np.roll(mask, 1, axis=1); up[:, 0] = mask[:, 0]
    down = mask & ~np.roll(mask, -1, axis=1); down[:, -1] = mask[:, -1]
    return float((left.sum() + right.sum()) * sy + (up.sum() + down.sum()) * sx)


def side(mask: np.ndarray, which: str, sx: float = 2.0, sy: float = 2.0) -> dict[str, float]:
    width = mask.shape[1]
    side_mask = mask[:, :width // 2] if which == "left" else mask[:, width // 2:]
    coords = np.argwhere(side_mask)
    area = float(len(coords) * sx * sy)
    perim = perimeter(side_mask, sx, sy)
    circularity = float(4 * np.pi * area / perim ** 2) if perim else np.nan
    if len(coords) >= 2:
        xy = coords.astype(float) * np.array([sy, sx])
        centered = xy - xy.mean(axis=0)
        eig = np.sort(np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(xy)), 0))[::-1]
        axes = 4 * np.sqrt(eig)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (axes[-1] / axes[0]) ** 2))) if axes[0] else 0.0
        minor = float(axes[-1])
    else:
        eccentricity = np.nan; minor = np.nan
    return {f"{which}_eccentricity": eccentricity, f"{which}_circularity": circularity, f"{which}_area_mm2": area, f"{which}_minor_axis_length_mm": minor, f"{which}_pixel_count": int(len(coords))}


def main():
    x = np.load(INPUT).astype(np.float32, copy=False)
    labels = pd.read_csv(ROWS)[["uid", "is_pathologic"]]
    if x.shape[0] != len(labels) or x.shape[1:] != (8, 109, 91):
        raise RuntimeError(f"unexpected input shape {x.shape}")
    mip = x.max(axis=1)
    rows = []
    for i, row in enumerate(labels.itertuples(index=False)):
        threshold = float(np.percentile(mip[i], 99.75))
        mask = (mip[i] >= threshold) & (mip[i] > 0)
        feature = {"uid": row.uid, "is_pathologic": int(row.is_pathologic), "mip_threshold_99_75": threshold, "mip_hot_pixels": int(mask.sum())}
        feature.update(side(mask, "left")); feature.update(side(mask, "right")); rows.append(feature)
    frame = pd.DataFrame(rows)
    oof = pd.read_csv(OOF)[["uid", "actual", "equal_ensemble_clipped"]]
    frame = frame.merge(oof, on="uid", how="left", validate="one_to_one")
    frame["oof_error"] = (frame.equal_ensemble_clipped >= 0.5).astype(int) != frame.actual
    summary = {"n_scans": len(frame), "input": "MIP across the same eight smoothed native slices used by the Gaussian 8-slice pipeline; output image size 109x91", "features": FEATURES, "definitions": {"area": "2D MIP top-0.25% pixel area in mm^2", "eccentricity": "2D PCA eccentricity sqrt(1-(minor/major)^2)", "circularity": "4*pi*area/perimeter^2", "minor_axis": "full 2D PCA minor-axis length in mm"}, "feature_results": []}
    for f in FEATURES:
        valid = frame[f].notna()
        label_auc = float(roc_auc_score(frame.loc[valid, "is_pathologic"], frame.loc[valid, f]))
        err_auc = float(roc_auc_score(frame.loc[valid, "oof_error"], frame.loc[valid, f]))
        a = frame.loc[valid & (frame.is_pathologic == 0), f].to_numpy(float); b = frame.loc[valid & (frame.is_pathologic == 1), f].to_numpy(float)
        e = frame.loc[valid & frame.oof_error, f].to_numpy(float); c = frame.loc[valid & ~frame.oof_error, f].to_numpy(float)
        summary["feature_results"].append({"feature": f, "healthy_mean": float(a.mean()), "disease_mean": float(b.mean()), "label_auc_direction_free": max(label_auc, 1-label_auc), "label_mann_whitney_p": float(stats.mannwhitneyu(a,b).pvalue), "error_auc_direction_free": max(err_auc, 1-err_auc), "error_mann_whitney_p": float(stats.mannwhitneyu(e,c).pvalue)})
    X = frame[FEATURES].to_numpy(float); y = frame.is_pathologic.to_numpy(int); pred = np.full(len(y), np.nan)
    for train, test in StratifiedKFold(5, shuffle=True, random_state=20260931).split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=20260931)); model.fit(X[train], y[train]); pred[test] = model.predict_proba(X[test])[:, 1]
    summary["feature_only_5fold_oof"] = {"auroc": float(roc_auc_score(y,pred)), "log_loss": float(log_loss(y,pred,labels=[0,1])), "accuracy": float(accuracy_score(y,pred>=.5)), "incorrect": int(np.sum((pred>=.5)!=y))}
    OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "mip_images.npy", mip); frame.to_csv(OUT / "mip_morphometric_features.csv", index=False); (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
