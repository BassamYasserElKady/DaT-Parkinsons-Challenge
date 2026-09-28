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
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results" / "cv5_submission_flip_clip" / "oof_predictions.csv"
VOLUMES = ROOT / "results" / "ants_rigid_brainmask_mode_normalized" / "volumes"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "healthy_mode_histogram_clusters"
N_BINS = 128
SEEDS = (20260914, 17, 41, 83, 131)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)
    healthy = labels.loc[labels["actual"] == 0, "uid"].sort_values().tolist()
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0

    rng = np.random.default_rng(20260914)
    samples = []
    positive_counts = []
    for uid in healthy:
        data = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        values = data[mask & np.isfinite(data) & (data > 0)]
        positive_counts.append(len(values))
        if len(values) > 10000:
            values = rng.choice(values, size=10000, replace=False)
        samples.append(values)
    sampled_log = np.log10(np.concatenate(samples))
    lower_log, upper_log = np.percentile(sampled_log, [0.1, 99.9])
    edges = np.linspace(float(lower_log), float(upper_log), N_BINS + 1)

    histograms = []
    for uid in healthy:
        data = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        values = data[mask & np.isfinite(data) & (data > 0)]
        log_values = np.clip(np.log10(values), lower_log, upper_log)
        counts, _ = np.histogram(log_values, bins=edges)
        distribution = counts.astype(np.float64) / counts.sum()
        histograms.append(distribution)
    histograms = np.asarray(histograms)

    # Square-root probabilities make Euclidean distance equivalent to Hellinger distance.
    hellinger = np.sqrt(histograms)
    pca = PCA(n_components=0.99, svd_solver="full")
    features = pca.fit_transform(hellinger)

    records = []
    assignments = {}
    for k in range(2, 30):
        seed_labels = []
        for seed in SEEDS:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            seed_labels.append(model.fit_predict(features))
        reference = seed_labels[0]
        stability = float(np.mean([
            adjusted_rand_score(a, b) for a, b in combinations(seed_labels, 2)
        ]))
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-6,
            n_init=5, random_state=SEEDS[0],
        ).fit(features)
        records.append({
            "clusters": k,
            "silhouette": float(silhouette_score(features, reference)),
            "davies_bouldin": float(davies_bouldin_score(features, reference)),
            "kmeans_stability_ari": stability,
            "gmm_bic": float(gmm.bic(features)),
            "smallest_cluster": int(np.bincount(reference).min()),
            "largest_cluster": int(np.bincount(reference).max()),
        })
        assignments[k] = reference
        print(f"finished k={k}", flush=True)

    scores = pd.DataFrame(records)
    best_silhouette = int(scores.loc[scores["silhouette"].idxmax(), "clusters"])
    best_bic = int(scores.loc[scores["gmm_bic"].idxmin(), "clusters"])
    stable = scores[scores["kmeans_stability_ari"] >= 0.90]
    best_stable = int(stable.loc[stable["silhouette"].idxmax(), "clusters"]) if len(stable) else best_silhouette

    scores.to_csv(OUT / "cluster_number_scores.csv", index=False)
    selected = pd.DataFrame({"uid": healthy, "cluster": assignments[best_stable]})
    selected.to_csv(OUT / "healthy_cluster_assignments.csv", index=False)
    summary = {
        "healthy_scans": len(healthy),
        "histogram_bins": N_BINS,
        "histogram_scale": "log10 intensity",
        "histogram_lower_intensity_p0_1": float(10 ** lower_log),
        "histogram_upper_intensity_p99_9": float(10 ** upper_log),
        "distance_representation": "square-root histogram probabilities (Hellinger geometry)",
        "pca_components_99_percent_variance": int(features.shape[1]),
        "best_silhouette_clusters": best_silhouette,
        "best_gmm_bic_clusters": best_bic,
        "recommended_stable_clusters": best_stable,
        "selected_cluster_sizes": selected["cluster"].value_counts().sort_index().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
