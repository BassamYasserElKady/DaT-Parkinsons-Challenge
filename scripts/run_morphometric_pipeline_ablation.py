#!/usr/bin/env python3
"""Ablate the article-inspired morphometric preprocessing stages."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--min-component-pixels", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def otsu_threshold(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0 or float(values.max()) <= float(values.min()):
        return float(values.max()) if values.size else 0.0
    hist, edges = np.histogram(values, bins=256, range=(float(values.min()), float(values.max())))
    weights = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centers)
    total = cumulative[-1]
    total_mean = cumulative_mean[-1]
    denominator = cumulative * (total - cumulative)
    numerator = (total_mean * cumulative - cumulative_mean) ** 2
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(centers[int(np.argmax(score))])


def largest_component(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    valid = np.flatnonzero(sizes >= min_size)
    valid = valid[valid != 0]
    if valid.size == 0:
        return np.zeros_like(mask, dtype=bool)
    component_id = int(valid[np.argmax(sizes[valid])])
    return labels == component_id


def binary_features(mask: np.ndarray, min_size: int, keep_largest: bool = True) -> tuple[float, ...]:
    if keep_largest:
        mask = largest_component(mask, min_size)
    else:
        mask = mask.astype(bool, copy=False)
    if not mask.any():
        return (0.0,) * 6
    coords = np.argwhere(mask)
    area = float(coords.shape[0])
    exposed = np.zeros_like(mask, dtype=np.int32)
    exposed[1:, :] += mask[1:, :] != mask[:-1, :]
    exposed[:-1, :] += mask[:-1, :] != mask[1:, :]
    exposed[:, 1:] += mask[:, 1:] != mask[:, :-1]
    exposed[:, :-1] += mask[:, :-1] != mask[:, 1:]
    perimeter = float(exposed[mask].sum())
    circularity = float(4.0 * np.pi * area / perimeter**2) if perimeter else 0.0
    centered = coords.astype(float) - coords.mean(axis=0, keepdims=True)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(centered.T @ centered / max(area, 1.0)), 0.0))[::-1]
    major = float(4.0 * np.sqrt(eigenvalues[0]))
    minor = float(4.0 * np.sqrt(eigenvalues[1]))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigenvalues[1] / eigenvalues[0]))) if eigenvalues[0] else 0.0
    return area, perimeter, circularity, major, minor, eccentricity


def soft_features(weights: np.ndarray) -> tuple[float, ...]:
    """Continuous analogue used only for the no-binary-mask ablation."""
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    mass = float(weights.sum())
    if mass <= 0:
        return (0.0,) * 6
    total_variation = float(
        np.abs(np.diff(weights, axis=0)).sum() + np.abs(np.diff(weights, axis=1)).sum()
    )
    circularity = float(4.0 * np.pi * mass / total_variation**2) if total_variation else 0.0
    yy, xx = np.indices(weights.shape, dtype=float)
    mean_y = float((weights * yy).sum() / mass)
    mean_x = float((weights * xx).sum() / mass)
    centered = np.stack((yy - mean_y, xx - mean_x), axis=-1)
    covariance = (centered * weights[..., None]).reshape(-1, 2).T @ centered.reshape(-1, 2) / mass
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    major = float(4.0 * np.sqrt(eigenvalues[0]))
    minor = float(4.0 * np.sqrt(eigenvalues[1]))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigenvalues[1] / eigenvalues[0]))) if eigenvalues[0] else 0.0
    return mass, total_variation, circularity, major, minor, eccentricity


def normalize_image(image: np.ndarray) -> np.ndarray:
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size == 0:
        return np.zeros_like(image, dtype=float)
    low, high = np.percentile(positive, [1, 99])
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def variant_features(projection: np.ndarray, variant: str, min_size: int) -> tuple[list[float], dict[str, float]]:
    sharpen = variant not in {"no_sharpening"}
    normalize = variant not in {"no_normalization"}
    adaptive = variant not in {"fixed_threshold"}
    split_hemispheres = variant not in {"no_hemisphere_split"}
    binary = variant not in {"soft_intensity"}
    component_filter = variant not in {"no_component_filter"}

    working = projection.astype(float)
    if sharpen:
        blurred = ndimage.gaussian_filter(working, sigma=1.0)
        working = np.maximum(working + 0.8 * (working - blurred), 0.0)
    if normalize:
        working = normalize_image(working)

    if adaptive:
        threshold = otsu_threshold(working)
    else:
        threshold = 0.5 if normalize else float(np.percentile(working[working > 0], 75))

    if binary:
        mask = (working >= threshold) & (working > 0)
        if component_filter:
            labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(labels.ravel())
            keep = np.flatnonzero(sizes >= min_size)
            keep = keep[keep != 0]
            mask = np.isin(labels, keep)
        if split_hemispheres:
            midline = mask.shape[0] // 2
            left = mask[:midline, :].T
            right = mask[midline:, :].T
            features = list(binary_features(left, min_size, component_filter) + binary_features(right, min_size, component_filter))
            names = [
                "left_area", "left_perimeter", "left_circularity", "left_major_axis", "left_minor_axis", "left_eccentricity",
                "right_area", "right_perimeter", "right_circularity", "right_major_axis", "right_minor_axis", "right_eccentricity",
            ]
        else:
            features = list(binary_features(mask.T, min_size, component_filter))
            names = ["global_area", "global_perimeter", "global_circularity", "global_major_axis", "global_minor_axis", "global_eccentricity"]
    else:
        weights = np.maximum(working, 0.0)
        if split_hemispheres:
            midline = weights.shape[0] // 2
            features = list(soft_features(weights[:midline, :].T) + soft_features(weights[midline:, :].T))
            names = [
                "left_soft_area", "left_soft_perimeter", "left_soft_circularity", "left_soft_major_axis", "left_soft_minor_axis", "left_soft_eccentricity",
                "right_soft_area", "right_soft_perimeter", "right_soft_circularity", "right_soft_major_axis", "right_soft_minor_axis", "right_soft_eccentricity",
            ]
        else:
            features = list(soft_features(weights.T))
            names = ["global_soft_area", "global_soft_perimeter", "global_soft_circularity", "global_soft_major_axis", "global_soft_minor_axis", "global_soft_eccentricity"]
    metadata = {"threshold": float(threshold), "hotspot_pixels": float(np.sum(working >= threshold))}
    return features, {"names": names, **metadata}


def extract_one(payload: tuple[str, str, int]) -> dict[str, object]:
    path, uid, min_size = payload
    image = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    data[~np.isfinite(data)] = 0.0
    data[data < 0] = 0.0
    z0 = min(38, max(0, data.shape[2] - 4))
    projection = np.max(data[:, :, z0:z0 + 4], axis=2)
    variants = ["full", "no_sharpening", "no_normalization", "fixed_threshold", "no_component_filter", "no_hemisphere_split", "soft_intensity"]
    result: dict[str, object] = {"uid": uid}
    for variant in variants:
        features, metadata = variant_features(projection, variant, min_size)
        result[f"{variant}__features"] = features
        result[f"{variant}__names"] = metadata["names"]
        result[f"{variant}__threshold"] = metadata["threshold"]
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/morphometric_pipeline_ablation"
    output.mkdir(parents=True, exist_ok=True)
    registered = root / "results/ants_registration_mni"
    with (root / "data/external/drive_inputs/train_labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = {row["uid"]: int(float(row["is_pathologic"])) for row in csv.DictReader(handle)}
    paths = sorted(registered.glob("*.nii.gz"))
    payloads = [(str(path), path.name.removesuffix(".nii.gz"), args.min_component_pixels) for path in paths if path.name.removesuffix(".nii.gz") in labels]

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(extract_one, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            row["is_pathologic"] = labels[str(row["uid"])]
            rows.append(row)
            if index % 100 == 0 or index == len(futures):
                print(f"features={index}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: str(row["uid"]))

    variants = ["full", "no_sharpening", "no_normalization", "fixed_threshold", "no_component_filter", "no_hemisphere_split", "soft_intensity"]
    feature_rows = []
    for row in rows:
        flat = {"uid": row["uid"], "is_pathologic": row["is_pathologic"]}
        for variant in variants:
            names = row[f"{variant}__names"]
            flat[f"{variant}__threshold"] = row[f"{variant}__threshold"]
            for name, value in zip(names, row[f"{variant}__features"]):
                flat[f"{variant}__{name}"] = value
        feature_rows.append(flat)
    feature_fieldnames = list(feature_rows[0])
    write_csv(output / "morphometric_ablation_features.csv", feature_rows, feature_fieldnames)

    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=int)
    uids = np.asarray([str(row["uid"]) for row in rows])
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.random_state)
    split_rows = [{"uid": uids[i], "split": "test" if i in set(test_idx) else "train", "is_pathologic": int(y[i])} for i in range(len(y))]
    write_csv(output / "split.csv", split_rows, ["uid", "split", "is_pathologic"])

    metric_rows = []
    for variant in variants:
        names = rows[0][f"{variant}__names"]
        X = np.asarray([[float(value) for value in row[f"{variant}__features"]] for row in rows], dtype=float)
        gamma = 1.0 / (2.0 * X.shape[1])
        model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma=gamma))
        model.fit(X[train_idx], y[train_idx])
        scores = model.decision_function(X[test_idx])
        predictions = (scores >= 0.0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[test_idx], predictions, labels=[0, 1]).ravel()
        cv = cross_validate(
            model, X[train_idx], y[train_idx], cv=StratifiedKFold(5, shuffle=True, random_state=args.random_state),
            scoring={"accuracy": "accuracy", "auroc": "roc_auc"}, n_jobs=args.jobs,
        )
        metric_rows.append({
            "variant": variant,
            "feature_count": X.shape[1],
            "feature_names": ";".join(names),
            "accuracy": accuracy_score(y[test_idx], predictions),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], predictions),
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": f1_score(y[test_idx], predictions),
            "auroc": roc_auc_score(y[test_idx], scores),
            "cv_accuracy_mean": cv["test_accuracy"].mean(),
            "cv_accuracy_std": cv["test_accuracy"].std(),
            "cv_auroc_mean": cv["test_auroc"].mean(),
            "cv_auroc_std": cv["test_auroc"].std(),
            "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp,
            "gamma": gamma, "C": 1.0,
        })
        prediction_rows = []
        for position, index in enumerate(test_idx):
            prediction_rows.append({"uid": uids[index], "actual": int(y[index]), "predicted": int(predictions[position]), "decision_score": float(scores[position])})
        write_csv(output / f"test_predictions_{variant}.csv", prediction_rows, ["uid", "actual", "predicted", "decision_score"])
    metric_rows.sort(key=lambda row: (-float(row["auroc"]), -float(row["accuracy"])))
    write_csv(output / "ablation_metrics.csv", metric_rows, list(metric_rows[0]))

    lines = [
        "# Morphometric preprocessing ablation",
        "",
        "All variants use the same 1,362 ANTs-registered scans, the same fixed four-slice MNI-space projection (one-based slices 39-42), the same stratified 70/30 split, and the same standardized medium-Gaussian RBF SVM.",
        "",
        "The full pipeline uses unsharp sharpening, robust min-max normalization, per-scan Otsu adaptive thresholding, 2D connected-component filtering, binary masks, and separate image-axis hemispheres. Binary area/perimeter/circularity require a mask; the soft-intensity variant is a continuous-moment alternative rather than an exact binary-mask ablation.",
        "",
        "## Results",
        "",
        "| Variant | Features | Accuracy | Balanced accuracy | AUROC | CV accuracy | CV AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(f"| {row['variant']} | {row['feature_count']} | {float(row['accuracy']):.4f} | {float(row['balanced_accuracy']):.4f} | {float(row['auroc']):.4f} | {float(row['cv_accuracy_mean']):.4f} | {float(row['cv_auroc_mean']):.4f} |")
    lines.extend([
        "",
        "The article does not specify its exact sharpening kernel, adaptive threshold formula, connectivity rule, or physical-unit convention, so this is an explicit operational ablation rather than an exact reproduction.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output={output}")
    print("ranked_results=")
    for row in metric_rows:
        print(f"{row['variant']}: accuracy={float(row['accuracy']):.4f}, auroc={float(row['auroc']):.4f}")


if __name__ == "__main__":
    main()
