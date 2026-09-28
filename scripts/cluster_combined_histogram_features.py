#!/usr/bin/env python3

import json
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "results" / "scan_clusters_10" / "feature_table.csv"
VOLUMES = ROOT / "results" / "ants_rigid_brainmask_mode_normalized" / "volumes"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "combined_histogram_clusters_10"
ORIGINAL_FEATURES = [
    "nonempty_z_slices", "positive_voxel_count", "positive_mean", "positive_std",
    "occipital_mean", "left_caudate_mean", "right_caudate_mean",
    "left_putamen_mean", "right_putamen_mean", "mean_caudate_sbr_atlas",
    "mean_putamen_sbr_atlas", "putamen_asymmetry_atlas", "caudate_asymmetry_atlas",
]
SEEDS = (20260914, 17, 41, 83, 131, 211, 307, 401, 503, 601, 701, 809)
WEIGHTS = (0.25, 0.5, 1.0, 2.0)
N_BINS = 128


def histogram_block(uids, mask):
    rng = np.random.default_rng(20260914)
    samples = []
    for uid in uids:
        a = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        v = a[mask & np.isfinite(a) & (a > 0)]
        samples.append(rng.choice(v, 10000, replace=False) if len(v) > 10000 else v)
    sample_log = np.log10(np.concatenate(samples))
    lo, hi = np.percentile(sample_log, [0.1, 99.9])
    edges = np.linspace(float(lo), float(hi), N_BINS + 1)
    rows = []
    for uid in uids:
        a = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        v = a[mask & np.isfinite(a) & (a > 0)]
        counts, _ = np.histogram(np.clip(np.log10(v), lo, hi), bins=edges)
        rows.append(counts / counts.sum())
    return np.asarray(rows), float(10**lo), float(10**hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(FEATURES_PATH)
    table = table.sort_values("uid").reset_index(drop=True)
    uids = table.uid.tolist()
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0

    original = StandardScaler().fit_transform(table[ORIGINAL_FEATURES].to_numpy(dtype=float))
    hist, lower, upper = histogram_block(uids, mask)
    hist_pca = PCA(n_components=0.99, svd_solver="full").fit_transform(np.sqrt(hist))
    hist_z = StandardScaler().fit_transform(hist_pca)
    np.savez_compressed(OUT / "combined_feature_blocks.npz", original=original, histogram_pca=hist_z, uids=np.asarray(uids))

    results = []
    best = None
    for weight in WEIGHTS:
        x = np.concatenate([original / np.sqrt(original.shape[1]), weight * hist_z / np.sqrt(hist_z.shape[1])], axis=1)
        assignments = []
        for seed in SEEDS:
            assignments.append(KMeans(n_clusters=10, n_init=20, random_state=seed).fit_predict(x))
        stability = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(assignments, 2)]))
        labels = assignments[0]
        silhouette = float(silhouette_score(x, labels))
        db = float(davies_bouldin_score(x, labels))
        sizes = np.bincount(labels, minlength=10)
        row = {"histogram_weight": weight, "stability_ari": stability, "silhouette": silhouette, "davies_bouldin": db, "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max())}
        results.append(row)
        candidate_key = (stability, silhouette)
        if best is None or candidate_key > best[0]:
            best = (candidate_key, weight, labels)

    scores = pd.DataFrame(results)
    scores.to_csv(OUT / "combined_cluster_scores.csv", index=False)
    _, best_weight, best_labels = best
    assignment = pd.DataFrame({"uid": uids, "gmm_cluster": best_labels, "actual": table.actual.to_numpy()})
    assignment.to_csv(OUT / "best_10_cluster_assignments.csv", index=False)
    cluster_sizes = assignment.groupby("gmm_cluster").size().sort_index().to_dict()
    summary = {
        "n_scans": len(uids), "n_clusters": 10, "original_feature_count": len(ORIGINAL_FEATURES),
        "histogram_bins": N_BINS, "histogram_pca_components": int(hist_z.shape[1]),
        "histogram_intensity_range": [lower, upper], "best_histogram_weight": best_weight,
        "best_solution": next(row for row in results if row["histogram_weight"] == best_weight),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
