#!/usr/bin/env python3
"""Evaluate axial versus multiplanar morphometrics fused with the 8-slice ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / "results/top025_full_geometry/masks"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OOF = ROOT / "results/cv5_real8_post_registration_gaussian/oof_predictions.csv"
OUT = ROOT / "results/multiplanar_morphometric_stack"
PLANES = {"axial": (2, (0, 1)), "coronal": (1, (0, 2)), "sagittal": (0, (1, 2))}
SIDES = ("left", "right")


def perimeter(mask: np.ndarray, spacing0: float, spacing1: float) -> float:
    a0 = mask & ~np.roll(mask, 1, axis=0); a0[0, :] = mask[0, :]
    a1 = mask & ~np.roll(mask, -1, axis=0); a1[-1, :] = mask[-1, :]
    b0 = mask & ~np.roll(mask, 1, axis=1); b0[:, 0] = mask[:, 0]
    b1 = mask & ~np.roll(mask, -1, axis=1); b1[:, -1] = mask[:, -1]
    return float((a0.sum() + a1.sum()) * spacing1 + (b0.sum() + b1.sum()) * spacing0)


def projected_features(mask: np.ndarray, affine: np.ndarray, axis: int, remaining: tuple[int, int], side: str) -> dict[str, float]:
    coords = np.argwhere(mask)
    world = nib.affines.apply_affine(affine, coords) if len(coords) else np.empty((0, 3))
    keep = world[:, 0] < 0 if side == "left" else world[:, 0] > 0
    side_mask = np.zeros_like(mask, dtype=bool)
    if len(coords):
        kept = coords[keep]
        side_mask[tuple(kept.T)] = True
    projection = np.any(side_mask, axis=axis)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    s0, s1 = float(spacing[remaining[0]]), float(spacing[remaining[1]])
    area = float(projection.sum() * s0 * s1)
    peri = perimeter(projection, s0, s1)
    circularity = float(4 * np.pi * area / peri**2) if peri > 0 else np.nan
    points = np.argwhere(projection).astype(float) * np.array([s0, s1])
    if len(points) >= 3:
        centered = points - points.mean(axis=0)
        eig = np.sort(np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(points)), 0))[::-1]
        axes = 4 * np.sqrt(eig)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (axes[-1] / axes[0]) ** 2))) if axes[0] > 0 else 0.0
        minor = float(axes[-1])
    else:
        eccentricity = minor = np.nan
    return {"eccentricity": eccentricity, "circularity": circularity, "area_mm2": area, "minor_axis_length_mm": minor}


def extract(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number, row in enumerate(labels.itertuples(index=False), 1):
        image = nib.as_closest_canonical(nib.load(str(MASKS / f"{row.uid}.nii.gz")))
        mask = np.asarray(image.dataobj).astype(bool)
        record = {"uid": row.uid, "is_pathologic": int(row.is_pathologic)}
        for plane, (axis, remaining) in PLANES.items():
            for side in SIDES:
                values = projected_features(mask, np.asarray(image.affine, float), axis, remaining, side)
                for name, value in values.items():
                    record[f"{plane}_{side}_{name}"] = value
        rows.append(record)
        if number % 200 == 0 or number == len(labels): print(f"processed={number}/{len(labels)}", flush=True)
    return pd.DataFrame(rows)


def metrics(y, p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "accuracy": float(accuracy_score(y, p >= 0.5)), "incorrect": int(np.sum((p >= 0.5) != y))}


def crossfit(frame: pd.DataFrame, features: list[str], y: np.ndarray, fold: np.ndarray, base_x: np.ndarray) -> np.ndarray:
    x = frame[features].to_numpy(float)
    out = np.full(len(y), np.nan)
    for current in sorted(np.unique(fold)):
        train, test = fold != current, fold == current
        scaler = StandardScaler().fit(x[train])
        model_x_train = np.column_stack([base_x[train], scaler.transform(x[train])])
        model_x_test = np.column_stack([base_x[test], scaler.transform(x[test])])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=20260931)
        model.fit(model_x_train, y[train]); out[test] = model.predict_proba(model_x_test)[:, 1]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)[["uid", "is_pathologic"]]
    features_frame = extract(labels)
    oof = pd.read_csv(OOF)[["uid", "actual", "fold", "resnet", "densenet", "equal_ensemble_clipped"]]
    frame = features_frame.merge(oof, on="uid", how="inner", validate="one_to_one")
    y = frame.actual.to_numpy(int); fold = frame.fold.to_numpy(int)
    feature_columns = [c for c in features_frame.columns if c not in {"uid", "is_pathologic"}]
    axial = [c for c in feature_columns if c.startswith("axial_")]
    all_planes = feature_columns
    base = np.column_stack([np.log(np.clip(frame.resnet, 1e-5, 1 - 1e-5) / np.clip(1 - frame.resnet, 1e-5, 1 - 1e-5)), np.log(np.clip(frame.densenet, 1e-5, 1 - 1e-5) / np.clip(1 - frame.densenet, 1e-5, 1 - 1e-5))])
    predictions = {"baseline": frame.equal_ensemble_clipped.to_numpy(float)}
    predictions["axial_morphometrics"] = crossfit(frame, axial, y, fold, base)
    predictions["multiplanar_morphometrics"] = crossfit(frame, all_planes, y, fold, base)
    summary = {"n_scans": len(frame), "feature_count_axial": len(axial), "feature_count_multiplanar": len(all_planes), "definitions": "Each plane uses left/right 2D projection of the registered top-0.25% mask; area is mm^2, circularity is 4*pi*A/P^2, eccentricity is PCA-based, and minor-axis length is the full PCA minor axis.", "results": {name: metrics(y, prediction) for name, prediction in predictions.items()}, "deltas_vs_baseline": {name: {"auroc": float(roc_auc_score(y, prediction) - roc_auc_score(y, predictions["baseline"])), "log_loss": float(log_loss(y, prediction, labels=[0, 1]) - log_loss(y, predictions["baseline"], labels=[0, 1])), "accuracy": float(accuracy_score(y, prediction >= 0.5) - accuracy_score(y, predictions["baseline"] >= 0.5))} for name, prediction in predictions.items() if name != "baseline"}}
    out = frame[["uid", "actual", "fold", "equal_ensemble_clipped", *feature_columns]].copy()
    out["axial_morphometric_probability"] = predictions["axial_morphometrics"]
    out["multiplanar_morphometric_probability"] = predictions["multiplanar_morphometrics"]
    out.to_csv(OUT / "oof_predictions.csv", index=False)
    features_frame.to_csv(OUT / "features.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
