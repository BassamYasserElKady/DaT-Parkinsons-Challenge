#!/usr/bin/env python3
"""Healthy-only mode-scaled histogram clusters with constrained Gamma mixtures."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results" / "cv5_submission_flip_clip" / "oof_predictions.csv"
VOLUMES = ROOT / "results" / "ants_rigid_brainmask" / "volumes"
MODE_NORMALIZED = ROOT / "results" / "ants_rigid_brainmask_mode_normalized" / "volumes"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "healthy_mode_scaled_gamma_clusters_10"
SEED = 20260914
N_CLUSTERS = 10
N_BINS = 128


def positive_values(uid, mask):
    path = MODE_NORMALIZED / f"{uid}.nii.gz"
    if not path.exists():
        path = VOLUMES / f"{uid}.nii.gz"
    a = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    v = a[mask & np.isfinite(a) & (a > 0)]
    return v


def gamma_logpdf(x, shape, scale):
    return (shape - 1) * np.log(x) - x / scale - gammaln(shape) - shape * np.log(scale)


def moments(x, w):
    n = max(float(w.sum()), 1e-8)
    mean = max(float(np.dot(w, x) / n), 1e-6)
    second = max(float(np.dot(w, x * x) / n), mean * mean + 1e-8)
    var = max(second - mean * mean, 1e-8)
    shape = float(np.clip(mean * mean / var, 1.05, 1e6))
    scale = float(max(mean / shape, 1e-8))
    return shape, scale


def constrained_fit(x, cluster):
    # Keep fitting bounded and deterministic for large voxel pools.
    note = "all positive mode-scaled in-mask voxels"
    if len(x) > 5_000_000:
        rng = np.random.default_rng(SEED + cluster)
        x = rng.choice(x, 5_000_000, replace=False)
        note = "deterministic 5M voxel cap"

    bg = x <= 1.1
    tail = x > 1.3
    if bg.sum() < 100:
        bg = x <= np.quantile(x, 0.9)
    if tail.sum() < 100:
        tail = x >= np.quantile(x, 0.9)
    s1, c1 = moments(x, bg.astype(float))
    s2, c2 = moments(x, tail.astype(float))
    w2 = float(np.clip(tail.mean(), 0.05, 0.10))
    w1 = 1.0 - w2
    previous = -np.inf
    for it in range(200):
        log_r = np.column_stack([
            np.log(w1) + gamma_logpdf(x, s1, c1),
            np.log(w2) + gamma_logpdf(x, s2, c2),
        ])
        m = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - m)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w2 = float(np.clip(nk[1] / len(x), 0.05, 0.10))
        w1 = 1.0 - w2
        s1, c1 = moments(x, r[:, 0])
        s2, c2 = moments(x, r[:, 1])
        # Background remains centered near the normalized mode; tail stays right-sided.
        mean1 = float(np.clip(s1 * c1, 0.90, 1.10))
        c1 = mean1 / s1
        mode1 = max((s1 - 1) * c1, 0.0)
        mode2 = max((s2 - 1) * c2, 0.0)
        if mode2 <= mode1 + 0.05:
            c2 = (mode1 + 0.05) / max(s2 - 1.0, 0.05)
        objective = float(np.sum(m[:, 0] + np.log(np.exp(log_r - m).sum(axis=1))))
        if abs(objective - previous) < 1e-5 * (1 + abs(objective)):
            break
        previous = objective
    mode1 = max((s1 - 1) * c1, 0.0)
    mode2 = max((s2 - 1) * c2, 0.0)
    components = []
    for k, (label, weight, shape, scale) in enumerate([
        ("background_uptake", w1, s1, c1),
        ("specific_striatal_uptake", w2, s2, c2),
    ]):
        components.append({
            "cluster": cluster, "component": k, "label": label,
            "mixture_weight": weight, "shape": shape, "scale": scale,
            "mean": shape * scale, "mode": max((shape - 1) * scale, 0.0),
            "assigned_voxels": int((r.argmax(axis=1) == k).sum()),
            "fit_voxels": len(x), "fit_iterations": it + 1,
            "fit_method": "constrained_VB_style_Gamma_mixture",
            "fit_note": note,
        })
    return components, objective, note


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0
    labels = pd.read_csv(LABELS)
    healthy = labels.loc[labels.actual == 0, "uid"].astype(str).sort_values().tolist()

    # Per-scan mode audit and histogram representation. The source volumes are already
    # mode-normalized; recompute the mode here to verify the invariant explicitly.
    values = {}
    mode_rows = []
    for uid in healthy:
        v = positive_values(uid, mask)
        mode = float(np.quantile(v, 0.5)) if len(v) == 0 else float(np.histogram(v, bins=256)[1][np.argmax(np.histogram(v, bins=256)[0])])
        values[uid] = v
        mode_rows.append({"uid": uid, "positive_voxels": len(v), "audit_mode": mode,
                          "median_scaled_intensity": float(np.median(v))})
    sample = np.concatenate([np.random.default_rng(SEED + i).choice(v, min(10000, len(v)), replace=False)
                             for i, v in enumerate(values.values())])
    lo, hi = np.percentile(np.log10(np.maximum(sample, 1e-8)), [0.1, 99.9])
    edges = np.linspace(lo, hi, N_BINS + 1)
    hists = []
    for uid in healthy:
        counts, _ = np.histogram(np.clip(np.log10(np.maximum(values[uid], 1e-8)), lo, hi), bins=edges)
        hists.append(counts / max(counts.sum(), 1))
    z = PCA(n_components=0.99, svd_solver="full", random_state=SEED).fit_transform(np.sqrt(np.asarray(hists)))
    assignments = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=SEED).fit_predict(z)
    assign = pd.DataFrame({"uid": healthy, "cluster": assignments})
    assign.to_csv(OUT / "healthy_cluster_assignments.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(OUT / "per_scan_mode_audit.csv", index=False)

    comps, summaries = [], []
    for cluster in range(N_CLUSTERS):
        uids = assign.loc[assign.cluster == cluster, "uid"].tolist()
        x = np.concatenate([values[u] for u in uids]) if uids else np.array([], dtype=np.float32)
        if len(x):
            fitted, objective, note = constrained_fit(x, cluster)
            comps.extend(fitted)
            summaries.append({"cluster": cluster, "healthy_scans": len(uids), "positive_voxels": len(x),
                              "fit_method": "constrained_VB_style_Gamma_mixture", "objective": objective,
                              "status": "completed", "fit_note": note})
        else:
            summaries.append({"cluster": cluster, "healthy_scans": 0, "positive_voxels": 0,
                              "fit_method": "", "objective": np.nan, "status": "no scans", "fit_note": ""})
        print(f"cluster={cluster} healthy_scans={len(uids)} voxels={len(x)}", flush=True)
    pd.DataFrame(comps).to_csv(OUT / "gamma_mixture_components.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "cluster_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "healthy_scans": len(healthy), "clusters": 10,
        "normalization": "per-scan positive brain-mask values divided by individual primary histogram mode",
        "clustering": "healthy controls only; square-root histogram PCA plus KMeans",
        "mixture": "two-component constrained VB-style Gamma mixture with Dirichlet-style responsibilities",
        "constraints": "background mean clipped to [0.90,1.10], weights background [0.90,0.95], striatal [0.05,0.10], striatal mode > background mode",
        "pd_assignment_rule": "match future PD scans using only scaled values <= 1.1",
        "paper_note": "This is a constrained VB-style implementation, not a line-by-line reproduction of the article's Laplace/Taylor shape posterior.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
