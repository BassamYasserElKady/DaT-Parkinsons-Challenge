#!/usr/bin/env python3
"""Refit fixed healthy histogram clusters with relaxed Gamma constraints."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "healthy_mode500_tail_gamma_clusters_10"
VOLUMES = BASE / "normalized_healthy_volumes"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "healthy_mode500_relaxed_gamma_clusters_10"
SEED = 20260914


def gamma_logpdf(x, shape, scale):
    return (shape - 1) * np.log(x) - x / scale - gammaln(shape) - shape * np.log(scale)


def moments(x, weights):
    n = max(float(weights.sum()), 1e-8)
    mean = max(float(np.dot(weights, x) / n), 1e-6)
    second = max(float(np.dot(weights, x * x) / n), mean * mean + 1e-8)
    var = max(second - mean * mean, 1e-8)
    shape = float(np.clip(mean * mean / var, 1.05, 1e5))
    return shape, float(max(mean / shape, 1e-8))


def fit_cluster(x, cluster):
    note = "all normalized voxels"
    if len(x) > 5_000_000:
        x = np.random.default_rng(SEED + cluster).choice(x, 5_000_000, replace=False)
        note = "deterministic 5M voxel cap"

    bg0 = x <= 1.25
    tail0 = x > 1.50
    if bg0.sum() < 100:
        bg0 = x <= np.quantile(x, 0.9)
    if tail0.sum() < 100:
        tail0 = x >= np.quantile(x, 0.98)
    s1, c1 = moments(x, bg0.astype(float))
    s2, c2 = moments(x, tail0.astype(float))
    # Initial proportions requested by the relaxed formulation.
    w1, w2 = 0.98, 0.02
    mean1 = float(np.clip(s1 * c1, 0.90, 1.10))
    c1 = mean1 / s1
    mean2_min = max(1.5, mean1 + 0.5)
    c2 = max(c2, mean2_min / s2)
    prev = -np.inf
    for it in range(250):
        log_r = np.column_stack([
            np.log(w1) + gamma_logpdf(x, s1, c1),
            np.log(w2) + gamma_logpdf(x, s2, c2),
        ])
        mx = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - mx)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w2 = float(np.clip(nk[1] / len(x), 0.015, 0.05))
        w1 = 1.0 - w2
        s1, c1 = moments(x, r[:, 0])
        s2, c2 = moments(x, r[:, 1])
        mean1 = float(np.clip(s1 * c1, 0.90, 1.10))
        c1 = mean1 / s1
        mean2_min = max(1.5, mean1 + 0.5)
        # Mean separation only; the Gamma mode remains free.
        c2 = max(c2, mean2_min / s2)
        objective = float(np.sum(mx[:, 0] + np.log(np.exp(log_r - mx).sum(axis=1))))
        if abs(objective - prev) < 1e-5 * (1 + abs(objective)):
            break
        prev = objective
    labels = r.argmax(axis=1)
    rows = []
    for k, (name, weight, shape, scale) in enumerate([
        ("background_uptake", w1, s1, c1),
        ("specific_striatal_uptake", w2, s2, c2),
    ]):
        rows.append({
            "cluster": cluster, "component": k, "label": name,
            "mixture_weight": weight, "shape": shape, "scale": scale,
            "mean": shape * scale, "mode": max((shape - 1) * scale, 0.0),
            "assigned_voxels": int((labels == k).sum()), "fit_voxels": len(x),
            "iterations": it + 1, "fit_method": "relaxed_mean_separation_Gamma_mixture",
            "fit_note": note,
        })
    return rows, objective


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0
    assignments = pd.read_csv(BASE / "healthy_cluster_assignments.csv")
    values = {}
    for uid in assignments.uid.astype(str):
        a = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        values[uid] = a[mask & np.isfinite(a) & (a > 0)]
    components, summaries = [], []
    for cluster in range(10):
        uids = assignments.loc[assignments.cluster == cluster, "uid"].astype(str).tolist()
        x = np.concatenate([values[u] for u in uids])
        rows, objective = fit_cluster(x, cluster)
        components.extend(rows)
        summaries.append({"cluster": cluster, "healthy_scans": len(uids),
                          "positive_voxels": len(x), "objective": objective,
                          "status": "completed"})
        print(f"cluster={cluster} healthy_scans={len(uids)} voxels={len(x)}", flush=True)
    assignments.to_csv(OUT / "healthy_cluster_assignments.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "gamma_mixture_components.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "cluster_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "healthy_scans": int(len(assignments)), "clusters": 10,
        "normalization_source": str(BASE / "normalized_healthy_volumes"),
        "pi2_bounds": [0.015, 0.05], "initial_weights": [0.98, 0.02],
        "background_initialization": "x <= 1.25",
        "striatal_initialization": "x > 1.50 or top 2 percent",
        "constraint": "mean2 >= max(1.5, mean1 + 0.5); Gamma mode unconstrained",
        "method": "moment-initialized constrained Gamma mixture",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
