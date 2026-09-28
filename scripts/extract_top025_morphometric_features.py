#!/usr/bin/env python3
"""Extract left/right morphometrics from registered top-0.25% voxel masks."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / "results/top025_full_geometry/masks"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OOF = ROOT / "results/cv5_real8_post_registration_gaussian/oof_predictions.csv"
OUT = ROOT / "results/top025_morphometric_features"
FEATURES = [
    "right_eccentricity",
    "left_eccentricity",
    "left_circularity",
    "right_circularity",
    "right_area_mm2",
    "left_area_mm2",
    "right_minor_axis_length_mm",
]


def perimeter_2d(mask: np.ndarray, sx: float, sy: float) -> float:
    """Approximate physical perimeter from exposed 4-neighbour pixel edges."""
    edge_x = mask & ~np.roll(mask, 1, axis=0)
    edge_x[0, :] = mask[0, :]
    edge_x |= mask & ~np.roll(mask, -1, axis=0)
    edge_x[-1, :] |= mask[-1, :]
    edge_y = mask & ~np.roll(mask, 1, axis=1)
    edge_y[:, 0] = mask[:, 0]
    edge_y |= mask & ~np.roll(mask, -1, axis=1)
    edge_y[:, -1] |= mask[:, -1]
    # Edges normal to x have length sy; edges normal to y have length sx.
    return float(np.count_nonzero(edge_x) * sy / 2 + np.count_nonzero(edge_y) * sx / 2)


def side_features(mask: np.ndarray, affine: np.ndarray, side: str) -> dict[str, float]:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = float(np.prod(spacing))
    coords = np.argwhere(mask)
    world = nib.affines.apply_affine(affine, coords) if len(coords) else np.empty((0, 3))
    # Geometric split is performed in template/world x coordinates.
    selected = world[:, 0] < 0 if side == "left" else world[:, 0] > 0
    coords = coords[selected]
    world = world[selected]
    projection = np.zeros(mask.shape[:2], dtype=bool)
    if len(coords):
        projection[coords[:, 0], coords[:, 1]] = True
    area_mm2 = float(projection.sum() * spacing[0] * spacing[1])
    perimeter = perimeter_2d(projection, float(spacing[0]), float(spacing[1]))
    circularity = float(4 * np.pi * area_mm2 / (perimeter * perimeter)) if perimeter > 0 else np.nan
    if len(world) >= 3:
        centered = world - world.mean(axis=0)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(world)), 0))[::-1]
        axes = 4 * np.sqrt(eigenvalues)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (axes[-1] / axes[0]) ** 2))) if axes[0] > 0 else 0.0
        minor_axis = float(axes[-1])
    else:
        eccentricity = np.nan
        minor_axis = np.nan
    return {
        f"{side}_eccentricity": eccentricity,
        f"{side}_circularity": circularity,
        f"{side}_area_mm2": area_mm2,
        f"{side}_volume_mm3": float(len(world) * voxel_volume),
        f"{side}_minor_axis_length_mm": minor_axis,
        f"{side}_voxel_count": int(len(world)),
    }


def main():
    labels = pd.read_csv(LABELS)[["uid", "is_pathologic"]]
    rows = []
    for number, row in enumerate(labels.itertuples(index=False), 1):
        image = nib.as_closest_canonical(nib.load(str(MASKS / f"{row.uid}.nii.gz")))
        mask = np.asarray(image.dataobj).astype(bool)
        feature = {"uid": row.uid, "is_pathologic": int(row.is_pathologic)}
        feature.update(side_features(mask, np.asarray(image.affine, dtype=float), "left"))
        feature.update(side_features(mask, np.asarray(image.affine, dtype=float), "right"))
        rows.append(feature)
        if number % 200 == 0 or number == len(labels):
            print(f"processed={number}/{len(labels)}", flush=True)
    frame = pd.DataFrame(rows)
    # Error comparison uses the completed, independent smoothed 8-slice OOF predictions.
    oof = pd.read_csv(OOF)[["uid", "actual", "equal_ensemble_clipped"]]
    frame = frame.merge(oof, on="uid", how="left", validate="one_to_one")
    frame["oof_error"] = (frame.equal_ensemble_clipped >= 0.5).astype(int) != frame.actual
    summary = {"n_scans": len(frame), "features": FEATURES, "definitions": {"area": "axial projection area in mm^2", "eccentricity": "3D PCA eccentricity sqrt(1-(minor/major)^2)", "circularity": "4*pi*projection area/perimeter^2", "minor_axis": "full 3D PCA minor-axis length in mm"}, "feature_results": []}
    for feature in FEATURES:
        valid = frame[[feature, "is_pathologic"]].notna().all(axis=1)
        healthy = frame.loc[valid & (frame.is_pathologic == 0), feature].to_numpy(float)
        disease = frame.loc[valid & (frame.is_pathologic == 1), feature].to_numpy(float)
        error = frame.loc[valid & frame.oof_error, feature].to_numpy(float)
        correct = frame.loc[valid & ~frame.oof_error, feature].to_numpy(float)
        auc = float(roc_auc_score(frame.loc[valid, "is_pathologic"], frame.loc[valid, feature]))
        u_label = stats.mannwhitneyu(healthy, disease, alternative="two-sided")
        u_error = stats.mannwhitneyu(error, correct, alternative="two-sided")
        error_auc = float(roc_auc_score(frame.loc[valid, "oof_error"], frame.loc[valid, feature]))
        summary["feature_results"].append({"feature": feature, "healthy_mean": float(healthy.mean()), "disease_mean": float(disease.mean()), "label_auc_directional": auc, "label_auc_direction_free": max(auc, 1 - auc), "label_mann_whitney_p": float(u_label.pvalue), "error_auc_directional": error_auc, "error_auc_direction_free": max(error_auc, 1 - error_auc), "error_mann_whitney_p": float(u_error.pvalue)})
    X = frame[FEATURES].to_numpy(float)
    y = frame.is_pathologic.to_numpy(int)
    predictions = np.full(len(frame), np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=20260931)
    for train, test in skf.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=20260931))
        model.fit(X[train], y[train]); predictions[test] = model.predict_proba(X[test])[:, 1]
    summary["feature_only_5fold_oof"] = {"auroc": float(roc_auc_score(y, predictions)), "log_loss": float(log_loss(y, predictions, labels=[0, 1])), "accuracy": float(accuracy_score(y, predictions >= 0.5)), "incorrect": int(np.sum((predictions >= 0.5) != y))}
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "morphometric_features.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
