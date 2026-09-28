#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import ConvexHull, cKDTree, distance
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / "results/ants_registration_mni"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OOF = ROOT / "results/cv5_submission_flip_clip/oof_predictions.csv"
OUT = ROOT / "results/top025_full_geometry"
MASK_DIR = OUT / "masks"


def safe_div(a, b):
    return float(a / b) if b else 0.0


def exposed_surface(mask, spacing):
    area = 0.0
    for axis in range(3):
        face_area = float(np.prod([spacing[i] for i in range(3) if i != axis]))
        area += int(np.count_nonzero(np.take(mask, 0, axis=axis))) * face_area
        area += int(np.count_nonzero(np.take(mask, -1, axis=axis))) * face_area
        area += int(np.count_nonzero(np.diff(mask.astype(np.int8), axis=axis))) * face_area
    return float(area)


def centroid(points):
    return points.mean(axis=0) if len(points) else np.full(3, np.nan)


def point_distance_metrics(a, b):
    if not len(a) or not len(b):
        return np.nan, np.nan, np.nan
    da = cKDTree(b).query(a, k=1, workers=1)[0]
    db = cKDTree(a).query(b, k=1, workers=1)[0]
    both = np.concatenate((da, db))
    return float(both.mean()), float(np.percentile(both, 95)), float(both.max())


def extract_one(payload):
    index, uid, label = payload
    image = nib.as_closest_canonical(nib.load(str(REGISTERED / f"{uid}.nii.gz")))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0).astype(np.float32)
    threshold = float(np.percentile(data, 99.75))
    mask = (data >= threshold) & (data > 0)
    coords = np.argwhere(mask)
    affine = np.asarray(image.affine, dtype=np.float64)
    world = nib.affines.apply_affine(affine, coords)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = float(np.prod(spacing))
    values = data[mask].astype(np.float64)
    n = len(coords)

    center = centroid(world)
    centered = world - center
    covariance = centered.T @ centered / max(n, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    axes = 4 * np.sqrt(eigenvalues)
    world_min, world_max = world.min(axis=0), world.max(axis=0)
    extents = world_max - world_min + spacing
    bbox_volume = float(np.prod(extents))
    physical_volume = n * voxel_volume
    surface = exposed_surface(mask, spacing)
    sphericity = safe_div(np.pi ** (1 / 3) * (6 * physical_volume) ** (2 / 3), surface)
    compactness = safe_div(36 * np.pi * physical_volume**2, surface**3)

    try:
        hull = ConvexHull(world)
        hull_volume, hull_area = float(hull.volume), float(hull.area)
    except Exception:
        hull_volume = hull_area = np.nan

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    components, component_count = ndimage.label(mask, structure=structure)
    component_sizes = np.sort(np.bincount(components.ravel())[1:])[::-1]
    largest = int(component_sizes[0]) if component_sizes.size else 0
    second = int(component_sizes[1]) if component_sizes.size > 1 else 0

    left = world[:, 0] < 0
    right = world[:, 0] > 0
    left_world, right_world = world[left], world[right]
    right_reflected = right_world.copy()
    right_reflected[:, 0] *= -1
    symmetry_assd, symmetry_hd95, symmetry_hdmax = point_distance_metrics(left_world, right_reflected)
    left_center, reflected_right_center = centroid(left_world), centroid(right_reflected)
    bilateral_center_distance = float(np.linalg.norm(left_center - reflected_right_center)) if len(left_world) and len(right_world) else np.nan

    inverse = np.linalg.inv(affine)
    reflected = world.copy()
    reflected[:, 0] *= -1
    reflected_voxels = np.rint(nib.affines.apply_affine(inverse, reflected)).astype(int)
    valid_reflected = np.all((reflected_voxels >= 0) & (reflected_voxels < np.asarray(mask.shape)), axis=1)
    reflected_voxels = reflected_voxels[valid_reflected]
    mirror = np.zeros_like(mask)
    mirror[tuple(reflected_voxels.T)] = True
    intersection = int(np.count_nonzero(mask & mirror))
    union = int(np.count_nonzero(mask | mirror))
    symmetry_dice = safe_div(2 * intersection, int(mask.sum()) + int(mirror.sum()))
    symmetry_jaccard = safe_div(intersection, union)

    weighted_center = np.average(world, axis=0, weights=values)
    weighted_centered = world - weighted_center
    weighted_cov = (weighted_centered * values[:, None]).T @ weighted_centered / values.sum()
    weighted_eigen = np.sort(np.maximum(np.linalg.eigvalsh(weighted_cov), 0))[::-1]

    features = {
        "uid": uid,
        "is_pathologic": int(label),
        "threshold_99_75": threshold,
        "hot_voxels": int(n),
        "hot_fraction": float(mask.mean()),
        "physical_volume_mm3": physical_volume,
        "surface_area_mm2": surface,
        "surface_to_volume": safe_div(surface, physical_volume),
        "sphericity": sphericity,
        "compactness": compactness,
        "convex_hull_volume_mm3": hull_volume,
        "convex_hull_area_mm2": hull_area,
        "solidity": safe_div(physical_volume, hull_volume),
        "bbox_volume_mm3": bbox_volume,
        "bbox_occupancy": safe_div(physical_volume, bbox_volume),
        "bbox_extent_x_mm": float(extents[0]),
        "bbox_extent_y_mm": float(extents[1]),
        "bbox_extent_z_mm": float(extents[2]),
        "centroid_x_mm": float(center[0]),
        "centroid_y_mm": float(center[1]),
        "centroid_z_mm": float(center[2]),
        "weighted_centroid_x_mm": float(weighted_center[0]),
        "weighted_centroid_y_mm": float(weighted_center[1]),
        "weighted_centroid_z_mm": float(weighted_center[2]),
        "principal_axis_1_mm": float(axes[0]),
        "principal_axis_2_mm": float(axes[1]),
        "principal_axis_3_mm": float(axes[2]),
        "elongation_axis1_axis2": safe_div(axes[0], axes[1]),
        "flatness_axis2_axis3": safe_div(axes[1], axes[2]),
        "anisotropy_axis1_axis3": safe_div(axes[0], axes[2]),
        "principal_direction_1_x": float(eigenvectors[0, 0]),
        "principal_direction_1_y": float(eigenvectors[1, 0]),
        "principal_direction_1_z": float(eigenvectors[2, 0]),
        "weighted_spread_axis1_mm": float(4 * np.sqrt(weighted_eigen[0])),
        "weighted_spread_axis2_mm": float(4 * np.sqrt(weighted_eigen[1])),
        "weighted_spread_axis3_mm": float(4 * np.sqrt(weighted_eigen[2])),
        "component_count_26": int(component_count),
        "largest_component_voxels": largest,
        "largest_component_fraction": safe_div(largest, n),
        "second_component_voxels": second,
        "second_to_largest_ratio": safe_div(second, largest),
        "left_voxels": int(left.sum()),
        "right_voxels": int(right.sum()),
        "left_right_asymmetry": safe_div(int(left.sum()) - int(right.sum()), int(left.sum()) + int(right.sum())),
        "left_fraction": safe_div(int(left.sum()), int(left.sum()) + int(right.sum())),
        "bilateral_centroid_distance_mm": bilateral_center_distance,
        "mirror_dice": symmetry_dice,
        "mirror_jaccard": symmetry_jaccard,
        "mirror_assd_mm": symmetry_assd,
        "mirror_hausdorff95_mm": symmetry_hd95,
        "mirror_hausdorff_max_mm": symmetry_hdmax,
        "hot_intensity_mean": float(values.mean()),
        "hot_intensity_std": float(values.std()),
        "hot_intensity_min": float(values.min()),
        "hot_intensity_max": float(values.max()),
        "hot_intensity_sum": float(values.sum()),
        "left_intensity_mean": float(values[left].mean()) if left.any() else np.nan,
        "right_intensity_mean": float(values[right].mean()) if right.any() else np.nan,
        "left_right_intensity_asymmetry": safe_div(float(values[left].sum()) - float(values[right].sum()), float(values[left].sum()) + float(values[right].sum())),
    }
    mask_image = nib.Nifti1Image(mask.astype(np.uint8), affine, image.header)
    mask_image.set_data_dtype(np.uint8)
    nib.save(mask_image, str(MASK_DIR / f"{uid}.nii.gz"))
    return index, features, np.flatnonzero(mask.ravel()).astype(np.int32), mask.shape, affine


def consensus_mask(index_lists, shape, target_count):
    frequency = np.zeros(int(np.prod(shape)), dtype=np.int32)
    for indices in index_lists:
        frequency[indices] += 1
    selected = np.argpartition(frequency, -target_count)[-target_count:]
    mask = np.zeros(frequency.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(shape), frequency.reshape(shape)


def overlap_metrics(mask, reference, spacing):
    intersection = int(np.count_nonzero(mask & reference))
    union = int(np.count_nonzero(mask | reference))
    dice = safe_div(2 * intersection, int(mask.sum()) + int(reference.sum()))
    jaccard = safe_div(intersection, union)
    a = np.argwhere(mask) * spacing
    b = np.argwhere(reference) * spacing
    assd, hd95, hdmax = point_distance_metrics(a, b)
    center_distance = float(np.linalg.norm(centroid(a) - centroid(b)))
    return dice, jaccard, assd, hd95, hdmax, center_distance


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def compare_groups(frame, group, group_name):
    rows = []
    excluded = {"is_pathologic", "oof_actual", "oof_prediction", "oof_correct", "oof_log_loss"}
    for feature in frame.select_dtypes(include=[np.number]).columns:
        if feature in excluded:
            continue
        valid = frame[[feature]].notna().to_numpy().ravel()
        a = frame.loc[valid & (group == 0), feature].to_numpy(dtype=float)
        b = frame.loc[valid & (group == 1), feature].to_numpy(dtype=float)
        if len(a) < 3 or len(b) < 3 or np.nanstd(np.concatenate((a, b))) == 0:
            continue
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
        effect = safe_div(b.mean() - a.mean(), pooled)
        _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = roc_auc_score(np.concatenate((np.zeros(len(a)), np.ones(len(b)))), np.concatenate((a, b)))
        rows.append({
            "comparison": group_name,
            "feature": feature,
            "group0_n": len(a), "group1_n": len(b),
            "group0_mean": a.mean(), "group1_mean": b.mean(),
            "group0_median": np.median(a), "group1_median": np.median(b),
            "cohens_d_group1_minus_group0": effect,
            "mannwhitney_p": p_value,
            "univariate_auc": auc,
            "discrimination_auc": max(auc, 1 - auc),
            "higher_group": 1 if b.mean() > a.mean() else 0,
        })
    if rows:
        q = fdr_bh([row["mannwhitney_p"] for row in rows])
        for row, value in zip(rows, q):
            row["fdr_q"] = value
    return pd.DataFrame(rows).sort_values(["fdr_q", "discrimination_auc"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    payloads = [(i, row["uid"], int(row["is_pathologic"])) for i, row in enumerate(rows)]
    features = [None] * len(rows)
    index_lists = [None] * len(rows)
    shape = affine = None
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(extract_one, payload) for payload in payloads]
        for count, future in enumerate(as_completed(futures), 1):
            index, row, indices, current_shape, current_affine = future.result()
            features[index] = row
            index_lists[index] = indices
            shape, affine = current_shape, current_affine
            if count % 100 == 0 or count == len(rows):
                print(f"extracted={count}/{len(rows)}", flush=True)

    labels = np.asarray([int(row["is_pathologic"]) for row in rows])
    target_count = int(np.median([len(indices) for indices in index_lists]))
    consensus = {}
    for name, selected in {
        "all": np.arange(len(rows)),
        "healthy": np.flatnonzero(labels == 0),
        "pathologic": np.flatnonzero(labels == 1),
    }.items():
        mask, frequency = consensus_mask([index_lists[i] for i in selected], shape, target_count)
        consensus[name] = mask
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(OUT / f"consensus_{name}.nii.gz"))
        nib.save(nib.Nifti1Image((frequency / len(selected)).astype(np.float32), affine), str(OUT / f"frequency_{name}.nii.gz"))

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    for index, row in enumerate(features):
        mask = np.zeros(int(np.prod(shape)), dtype=bool)
        mask[index_lists[index]] = True
        mask = mask.reshape(shape)
        for name, reference in consensus.items():
            values = overlap_metrics(mask, reference, spacing)
            for suffix, value in zip(("dice", "jaccard", "assd_mm", "hausdorff95_mm", "hausdorff_max_mm", "centroid_distance_mm"), values):
                row[f"to_{name}_consensus_{suffix}"] = value
        if (index + 1) % 100 == 0 or index + 1 == len(features):
            print(f"consensus_compared={index + 1}/{len(features)}", flush=True)

    frame = pd.DataFrame(features)
    oof = pd.read_csv(OOF).rename(columns={"actual": "oof_actual", "prediction": "oof_prediction", "probability": "oof_probability"})
    frame = frame.merge(oof[["uid", "oof_actual", "oof_prediction", "oof_probability", "fold"]], on="uid", how="left")
    frame["oof_correct"] = (frame["oof_actual"] == frame["oof_prediction"]).astype(int)
    p = np.clip(frame["oof_probability"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    y = frame["oof_actual"].to_numpy(dtype=int)
    frame["oof_log_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    frame.to_csv(OUT / "geometry_features.csv", index=False)

    label_comparison = compare_groups(frame, frame["is_pathologic"].to_numpy(), "healthy_vs_pathologic")
    label_comparison.to_csv(OUT / "label_comparison.csv", index=False)
    error_group = (frame["oof_correct"].to_numpy() == 0).astype(int)
    error_comparison = compare_groups(frame, error_group, "correct_vs_incorrect")
    error_comparison.to_csv(OUT / "model_error_comparison.csv", index=False)

    geometry_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"is_pathologic", "oof_actual", "oof_prediction", "oof_probability", "oof_correct", "oof_log_loss", "fold"}
    ]
    correlation = frame[geometry_columns].corr(method="spearman")
    correlation.to_csv(OUT / "geometry_spearman_correlations.csv")

    matrix = csr_matrix(
        (
            np.ones(sum(map(len, index_lists)), dtype=np.float32),
            np.concatenate(index_lists),
            np.concatenate(([0], np.cumsum([len(item) for item in index_lists]))),
        ),
        shape=(len(rows), int(np.prod(shape))),
    )
    intersections = (matrix @ matrix.T).toarray()
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    dice_matrix = 2 * intersections / (counts[:, None] + counts[None, :])
    jaccard_matrix = intersections / (counts[:, None] + counts[None, :] - intersections)
    centroids = frame[["centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]].to_numpy(dtype=float)
    centroid_distance = distance.squareform(distance.pdist(centroids))
    standardized = StandardScaler().fit_transform(frame[geometry_columns].fillna(frame[geometry_columns].median()).to_numpy(dtype=float))
    feature_distance = distance.squareform(distance.pdist(standardized))
    np.savez_compressed(
        OUT / "pairwise_geometry_matrices.npz",
        uids=frame["uid"].to_numpy(), labels=labels,
        dice=dice_matrix.astype(np.float32), jaccard=jaccard_matrix.astype(np.float32),
        centroid_distance_mm=centroid_distance.astype(np.float32),
        standardized_feature_distance=feature_distance.astype(np.float32),
    )

    np.fill_diagonal(dice_matrix, -np.inf)
    np.fill_diagonal(feature_distance, np.inf)
    nearest_dice = np.argmax(dice_matrix, axis=1)
    nearest_geometry = np.argmin(feature_distance, axis=1)
    upper = np.triu_indices(len(rows), 1)
    same = labels[upper[0]] == labels[upper[1]]
    pair_summary = {
        "pair_count": int(len(same)),
        "same_label_pairs": int(same.sum()),
        "different_label_pairs": int((~same).sum()),
        "mean_dice_same_label": float(dice_matrix[upper][same].mean()),
        "mean_dice_different_label": float(dice_matrix[upper][~same].mean()),
        "mean_centroid_distance_same_label_mm": float(centroid_distance[upper][same].mean()),
        "mean_centroid_distance_different_label_mm": float(centroid_distance[upper][~same].mean()),
        "nearest_dice_label_agreement": float(np.mean(labels[nearest_dice] == labels)),
        "nearest_geometry_label_agreement": float(np.mean(labels[nearest_geometry] == labels)),
    }
    summary = {
        "scan_count": len(rows),
        "mask_definition": "full-volume nonnegative voxels at or above each scan's 99.75th percentile",
        "grid_shape": list(shape),
        "spacing_mm": spacing.tolist(),
        "median_hot_voxels": target_count,
        "geometry_feature_count": len(geometry_columns),
        "significant_label_features_fdr_005": int((label_comparison["fdr_q"] < 0.05).sum()),
        "significant_error_features_fdr_005": int((error_comparison["fdr_q"] < 0.05).sum()),
        "top_label_features": label_comparison.head(15).to_dict(orient="records"),
        "top_error_features": error_comparison.head(15).to_dict(orient="records"),
        "pairwise": pair_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Top-0.25% full-volume geometry analysis", "",
        f"Processed {len(rows)} MNI-registered scans on a {shape} grid with {spacing.tolist()} mm spacing.",
        f"Each extracted mask contains a median of {target_count} voxels.", "",
        f"Geometry features significantly associated with diagnosis after FDR correction: {summary['significant_label_features_fdr_005']}.",
        f"Geometry features significantly associated with OOF model error after FDR correction: {summary['significant_error_features_fdr_005']}.", "",
        "See geometry_features.csv for per-scan measurements, label_comparison.csv and model_error_comparison.csv for group statistics, pairwise_geometry_matrices.npz for all-pair comparisons, and masks/ for the extracted NIfTI masks.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output={OUT}", flush=True)


if __name__ == "__main__":
    main()
