#!/usr/bin/env python3
"""Classify labels from morphometry of registered top-0.25% hotspot masks."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURES = [
    "left_area",
    "left_perimeter",
    "left_circularity",
    "left_major_axis_length",
    "left_minor_axis_length",
    "left_eccentricity",
    "right_area",
    "right_perimeter",
    "right_circularity",
    "right_major_axis_length",
    "right_minor_axis_length",
    "right_eccentricity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--min-component-voxels", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def component_features(mask: np.ndarray) -> tuple[float, ...]:
    """Measure the largest 8-connected component in a 2D hemisphere mask."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return (0.0,) * 6
    sizes = np.bincount(labels.ravel())
    component_id = int(np.argmax(sizes[1:]) + 1)
    coords = np.argwhere(labels == component_id)
    area = float(coords.shape[0])
    component = labels == component_id

    # Four-neighbour exposed edges provide a stable pixel perimeter estimate.
    exposed = np.zeros_like(component, dtype=np.int32)
    exposed[1:, :] += component[1:, :] != component[:-1, :]
    exposed[:-1, :] += component[:-1, :] != component[1:, :]
    exposed[:, 1:] += component[:, 1:] != component[:, :-1]
    exposed[:, :-1] += component[:, :-1] != component[:, 1:]
    perimeter = float(exposed[component].sum())
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0

    centered = coords.astype(float) - coords.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(area, 1.0)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    major = float(4.0 * np.sqrt(eigenvalues[0]))
    minor = float(4.0 * np.sqrt(eigenvalues[1]))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigenvalues[1] / eigenvalues[0]))) if eigenvalues[0] else 0.0
    return area, perimeter, circularity, major, minor, eccentricity


def extract_one(payload: tuple[str, str, int]) -> dict[str, object]:
    path, uid, min_component_voxels = payload
    image = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    finite = np.isfinite(data)
    values = data[finite]
    threshold = float(np.percentile(values, 99.75))
    hotspot = finite & (data >= threshold)

    labels, n_components = ndimage.label(hotspot, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= min_component_voxels)
    keep = keep[keep != 0]
    filtered = np.isin(labels, keep)

    # Project the filtered hotspot mask through z, then take one component per side.
    projected = np.any(filtered, axis=2)
    midline = projected.shape[0] // 2
    left = projected[:midline, :].T
    right = projected[midline:, :].T
    left_features = component_features(left)
    right_features = component_features(right)
    feature_values = left_features + right_features
    return {
        "uid": uid,
        "threshold_99_75": threshold,
        "hotspot_voxels": int(hotspot.sum()),
        "filtered_hotspot_voxels": int(filtered.sum()),
        "components_before_filter": int(n_components),
        "components_kept": int(len(keep)),
        "left_projected_hotspots": int(left.sum()),
        "right_projected_hotspots": int(right.sum()),
        **dict(zip(FEATURES, feature_values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/ants_registered_top025_morphometric_medium_gaussian_svm"
    output.mkdir(parents=True, exist_ok=True)
    registered = root / "results/ants_registration_mni"
    labels_path = root / "data/external/drive_inputs/train_labels.csv"

    with labels_path.open(newline="", encoding="utf-8") as handle:
        labels = {row["uid"]: int(float(row["is_pathologic"])) for row in csv.DictReader(handle)}
    paths = sorted(registered.glob("*.nii.gz"))
    payloads = [(str(path), path.name.removesuffix(".nii.gz"), args.min_component_voxels) for path in paths if path.name.removesuffix(".nii.gz") in labels]

    feature_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(extract_one, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            row["is_pathologic"] = labels[str(row["uid"])]
            feature_rows.append(row)
            if index % 100 == 0 or index == len(futures):
                print(f"features={index}/{len(futures)}", flush=True)
    feature_rows.sort(key=lambda row: str(row["uid"]))

    metadata = [
        "uid", "is_pathologic", "threshold_99_75", "hotspot_voxels",
        "filtered_hotspot_voxels", "components_before_filter", "components_kept",
        "left_projected_hotspots", "right_projected_hotspots",
    ]
    write_csv(output / "morphometric_features.csv", feature_rows, metadata + FEATURES)

    X = np.asarray([[float(row[name]) for name in FEATURES] for row in feature_rows], dtype=float)
    y = np.asarray([int(row["is_pathologic"]) for row in feature_rows], dtype=int)
    uids = np.asarray([str(row["uid"]) for row in feature_rows])
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # MATLAB's "Medium Gaussian SVM" uses an RBF kernel with a medium scale.
    # For P standardized predictors, sigma=sqrt(P), hence gamma=1/(2P).
    gamma = 1.0 / (2.0 * len(FEATURES))
    model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma=gamma))
    model.fit(X[train_idx], y[train_idx])
    scores = model.decision_function(X[test_idx])
    predictions = (scores >= 0.0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[test_idx], predictions, labels=[0, 1]).ravel()
    metrics = {
        "model": "medium_gaussian_svm_rbf",
        "n_scans": len(y),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "patients_total": int(y.sum()),
        "healthy_total": int((y == 0).sum()),
        "patients_test": int(y[test_idx].sum()),
        "healthy_test": int((y[test_idx] == 0).sum()),
        "accuracy": float(accuracy_score(y[test_idx], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y[test_idx], predictions)),
        "auroc": float(roc_auc_score(y[test_idx], scores)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "gamma": gamma, "C": 1.0,
    }
    write_csv(output / "test_metrics.csv", [metrics], list(metrics))

    prediction_rows = []
    test_set = set(test_idx.tolist())
    for idx in test_idx:
        prediction_rows.append({
            "uid": uids[idx], "actual": int(y[idx]), "predicted": int(predictions[list(test_idx).index(idx)]),
            "decision_score": float(scores[list(test_idx).index(idx)]),
        })
    write_csv(output / "test_predictions.csv", prediction_rows, ["uid", "actual", "predicted", "decision_score"])

    cv = cross_validate(
        model, X[train_idx], y[train_idx], cv=StratifiedKFold(5, shuffle=True, random_state=args.random_state),
        scoring={"accuracy": "accuracy", "auroc": "roc_auc"}, n_jobs=args.jobs,
    )
    cv_metrics = {
        "cv_accuracy_mean": float(cv["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv["test_accuracy"].std()),
        "cv_auroc_mean": float(cv["test_auroc"].mean()),
        "cv_auroc_std": float(cv["test_auroc"].std()),
    }
    write_csv(output / "cross_validation_metrics.csv", [cv_metrics], list(cv_metrics))

    report = output / "report.md"
    report.write_text(
        "# Registered top-0.25% morphometric SVM\n\n"
        f"- Scans: {len(y)} ANTs-registered volumes\n"
        f"- Mask: voxels at or above the per-scan 99.75th percentile\n"
        f"- Connected-component filter: 3D 26-connectivity, minimum {args.min_component_voxels} voxels\n"
        "- Morphometry: largest 2D connected component per image-axis hemisphere after z-projection\n"
        "- Features: area, perimeter, circularity, major axis, minor axis, eccentricity per side (12 total)\n"
        f"- Split: stratified {1 - args.test_size:.0%}/{args.test_size:.0%}, random_state={args.random_state}\n"
        "- Classifier: standardized RBF SVM, C=1, gamma=1/(2*12), medium-Gaussian analogue\n\n"
        "## Held-out test\n\n"
        + "\n".join(f"- **{key}:** {value:.4f}" if isinstance(value, float) else f"- **{key}:** {value}" for key, value in metrics.items())
        + "\n\n## Training cross-validation\n\n"
        + "\n".join(f"- **{key}:** {value:.4f}" for key, value in cv_metrics.items())
        + "\n\nThis is an exploratory intensity-derived mask, not a ground-truth anatomical striatal segmentation.\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"test_metrics={metrics}")
    print(f"cv_metrics={cv_metrics}")


if __name__ == "__main__":
    main()
