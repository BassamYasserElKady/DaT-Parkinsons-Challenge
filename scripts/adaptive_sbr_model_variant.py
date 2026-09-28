#!/usr/bin/env python3
"""Adaptive contour SBR features and a small 70/30 patient classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.spatial import ConvexHull
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LEFT_LABELS = (5, 6)
RIGHT_LABELS = (16, 17)
OCCIPITAL_LABELS = (22, 23, 39, 40, 48)
FEATURES = [
    "adaptive_left_sbr",
    "adaptive_right_sbr",
    "adaptive_mean_sbr",
    "adaptive_min_sbr",
    "adaptive_striatal_asymmetry",
    "left_contour_area_voxels",
    "right_contour_area_voxels",
]


def top_points(values: np.ndarray, mask: np.ndarray, percent: float = 1.0) -> np.ndarray:
    flat_values = values[mask]
    flat_indices = np.flatnonzero(mask)
    count = max(3, int(np.ceil(flat_values.size * percent / 100.0)))
    selected = flat_indices[np.argpartition(flat_values, -count)[-count:]]
    return np.column_stack(np.unravel_index(selected, mask.shape))


def filled_hull(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    points = np.unique(points, axis=0)
    if len(points) < 3:
        result[points[:, 0], points[:, 1]] = True
        return result
    xy = points[:, [1, 0]]
    hull = ConvexHull(xy, qhull_options="QJ")
    polygon = xy[hull.vertices]
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    result[:] = PolygonPath(polygon).contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(shape)
    if not result.any():
        result[points[:, 0], points[:, 1]] = True
    return result


def adaptive_features(data: np.ndarray, subcortical: np.ndarray, cortical: np.ndarray, mask_method: str) -> dict[str, float | int]:
    left = np.isin(subcortical, LEFT_LABELS)
    right = np.isin(subcortical, RIGHT_LABELS)
    striatum = left | right
    reference = np.isin(cortical, OCCIPITAL_LABELS)
    slice_scores: list[float] = []
    for z in range(data.shape[2]):
        points = top_points(data[:, :, z], striatum[:, :, z])
        slice_scores.append(float(data[points[:, 0], points[:, 1], z].sum()))
    _, start = max((sum(slice_scores[z : z + 7]), z) for z in range(data.shape[2] - 6))
    zs = range(start, start + 7)

    target_masks = {}
    areas = {}
    means = {}
    for name, side in (("left", left), ("right", right)):
        mask3d = np.zeros(data.shape, dtype=bool)
        if mask_method == "global_top1":
            side_values = data[:, :, list(zs)][side[:, :, list(zs)]]
            count = max(3, int(np.ceil(side_values.size * 0.01)))
            selected_flat = np.flatnonzero(side[:, :, list(zs)])
            selected_flat = selected_flat[np.argpartition(side_values, -count)[-count:]]
            selected = np.column_stack(np.unravel_index(selected_flat, side[:, :, list(zs)].shape))
            for z in range(start, start + 7):
                points = selected[selected[:, 2] == z - start, :2]
                if len(points):
                    mask3d[:, :, z] = filled_hull((data.shape[0], data.shape[1]), points)
        elif mask_method == "slice_top1":
            for z in range(start, start + 7):
                points = top_points(data[:, :, z], side[:, :, z], 1.0)
                mask3d[:, :, z] = filled_hull((data.shape[0], data.shape[1]), points)
        else:
            raise ValueError(f"Unknown mask method: {mask_method}")
        target_masks[name] = mask3d
        areas[name] = int(mask3d.sum())
        means[name] = float(data[mask3d].mean()) if mask3d.any() else float("nan")

    reference_mean = float(data[reference].mean()) if reference.any() else float("nan")
    left_sbr = (means["left"] - reference_mean) / reference_mean if reference_mean > 0 else float("nan")
    right_sbr = (means["right"] - reference_mean) / reference_mean if reference_mean > 0 else float("nan")
    mean_sbr = float(np.nanmean([left_sbr, right_sbr]))
    denominator = means["left"] + means["right"]
    asymmetry = 2.0 * abs(means["left"] - means["right"]) / denominator if denominator else float("nan")
    return {
        "slice_start": start,
        "slice_end_exclusive": start + 7,
        "adaptive_reference_mean": reference_mean,
        "adaptive_left_target_mean": means["left"],
        "adaptive_right_target_mean": means["right"],
        "adaptive_left_sbr": left_sbr,
        "adaptive_right_sbr": right_sbr,
        "adaptive_mean_sbr": mean_sbr,
        "adaptive_min_sbr": float(np.nanmin([left_sbr, right_sbr])),
        "adaptive_striatal_asymmetry": asymmetry,
        "left_contour_area_voxels": areas["left"],
        "right_contour_area_voxels": areas["right"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--subcortical-atlas", type=Path, required=True)
    parser.add_argument("--cortical-atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-method", choices=("global_top1", "slice_top1"), default="global_top1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = nib.load(str(args.subcortical_atlas))
    subcortical = np.asarray(template.dataobj)
    cortical = np.asarray(nib.load(str(args.cortical_atlas)).dataobj)
    with args.labels.open(newline="", encoding="utf-8") as handle:
        labels = {row["uid"]: int(float(row["is_pathologic"])) for row in csv.DictReader(handle)}

    rows = []
    for path in sorted(args.registered_dir.glob("*.nii.gz")):
        uid = path.name.removesuffix(".nii.gz")
        data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
        row = {"uid": uid, "is_pathologic": labels[uid], "registered_image": str(path)}
        row.update(adaptive_features(np.clip(data, 0, None), subcortical, cortical, args.mask_method))
        rows.append(row)

    feature_path = args.output_dir / "adaptive_sbr_features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if all(np.isfinite(float(row[name])) for name in FEATURES)]
    X = np.asarray([[float(row[name]) for name in FEATURES] for row in valid])
    y = np.asarray([int(row["is_pathologic"]) for row in valid])
    indices = np.arange(len(valid))
    train_idx, test_idx = train_test_split(indices, test_size=0.30, stratify=y, random_state=args.seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=args.seed))
    model.fit(X[train_idx], y[train_idx])
    probabilities = model.predict_proba(X[test_idx])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[test_idx], predictions, labels=[0, 1]).ravel()
    metrics = {
        "total_n": len(valid),
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "test_patients": int(y[test_idx].sum()),
        "test_healthy": int((y[test_idx] == 0).sum()),
        "accuracy": accuracy_score(y[test_idx], predictions),
        "balanced_accuracy": balanced_accuracy_score(y[test_idx], predictions),
        "sensitivity": recall_score(y[test_idx], predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": precision_score(y[test_idx], predictions, zero_division=0),
        "f1": f1_score(y[test_idx], predictions, zero_division=0),
        "roc_auc": roc_auc_score(y[test_idx], probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    with (args.output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "is_pathologic", "probability", "prediction", "split"])
        writer.writeheader()
        for i, probability, prediction in zip(test_idx, probabilities, predictions):
            writer.writerow({"uid": valid[i]["uid"], "is_pathologic": y[i], "probability": probability, "prediction": prediction, "split": "test"})
    coefficients = model[-1].coef_[0]
    with (args.output_dir / "model_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "standardized_coefficient"])
        writer.writeheader()
        writer.writerows({"feature": name, "standardized_coefficient": value} for name, value in zip(FEATURES, coefficients))
    (args.output_dir / "report.md").write_text(
        f"# Adaptive SBR model on 184 scans ({args.mask_method})\n\n"
        f"Adaptive target regions used the {args.mask_method} strategy within bilateral striatal atlas search regions over the highest-scoring contiguous seven-slice window. The classifier is a standardized logistic regression with a stratified 70/30 split (`random_state=42`).\n\n"
        f"- Valid scans: {metrics['total_n']}\n- Train scans: {metrics['train_n']}\n- Held-out test scans: {metrics['test_n']}\n- Test accuracy: {metrics['accuracy']:.4f}\n- Test balanced accuracy: {metrics['balanced_accuracy']:.4f}\n- Test sensitivity: {metrics['sensitivity']:.4f}\n- Test specificity: {metrics['specificity']:.4f}\n- Test ROC AUC: {metrics['roc_auc']:.4f}\n",
        encoding="utf-8",
    )
    print(f"features={feature_path}")
    print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
