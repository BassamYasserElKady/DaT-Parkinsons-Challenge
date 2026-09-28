#!/usr/bin/env python3
"""Healthy-only 500-bin mode scaling and constrained two-Gamma mixtures."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "ants_registration_mni"
LABELS = ROOT / "results" / "cv5_submission_flip_clip" / "oof_predictions.csv"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "healthy_mode500_tail_gamma_clusters_10"
NORMALIZED = OUT / "normalized_healthy_volumes"
SEED = 20260914


def histogram_mode(x, bins=500):
    counts, edges = np.histogram(x[x > 0], bins=bins)
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2.0), int(i), int(counts[i])


def gamma_logpdf(x, shape, scale):
    return (shape - 1) * np.log(x) - x / scale - gammaln(shape) - shape * np.log(scale)


def moments(x, weights):
    n = max(float(weights.sum()), 1e-8)
    mean = max(float(np.dot(weights, x) / n), 1e-6)
    second = max(float(np.dot(weights, x * x) / n), mean * mean + 1e-8)
    var = max(second - mean * mean, 1e-8)
    shape = float(np.clip(mean * mean / var, 1.05, 1e5))
    return shape, float(max(mean / shape, 1e-8))


def enforce_parameters(shape, scale, minimum_mode=None, mean_bounds=None):
    if mean_bounds is not None:
        mean = float(np.clip(shape * scale, *mean_bounds))
        scale = mean / shape
    if minimum_mode is not None:
        scale = max(scale, float(minimum_mode) / max(shape - 1.0, 0.05))
    return shape, scale


def fit_cluster(x, cluster):
    note = "all positive mode-scaled voxels"
    if len(x) > 5_000_000:
        x = np.random.default_rng(SEED + cluster).choice(x, 5_000_000, replace=False)
        note = "deterministic 5M voxel cap"

    bg0 = x <= 1.3
    tail0 = x > 1.5
    if tail0.sum() < 100:
        tail0 = x >= np.quantile(x, 0.95)
    if bg0.sum() < 100:
        bg0 = x <= np.quantile(x, 0.9)
    s1, c1 = moments(x, bg0.astype(float))
    s2, c2 = moments(x, tail0.astype(float))
    s1, c1 = enforce_parameters(s1, c1, mean_bounds=(0.90, 1.10))
    s2, c2 = enforce_parameters(s2, c2, minimum_mode=1.5)
    w1, w2 = 0.93, 0.07
    prev = -np.inf
    for it in range(200):
        log_r = np.column_stack([
            np.log(w1) + gamma_logpdf(x, s1, c1),
            np.log(w2) + gamma_logpdf(x, s2, c2),
        ])
        mx = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - mx)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w2 = float(np.clip(nk[1] / len(x), 0.05, 0.10))
        w1 = 1.0 - w2
        s1, c1 = moments(x, r[:, 0])
        s2, c2 = moments(x, r[:, 1])
        s1, c1 = enforce_parameters(s1, c1, mean_bounds=(0.90, 1.10))
        s2, c2 = enforce_parameters(s2, c2, minimum_mode=1.5)
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
            "iterations": it + 1, "fit_method": "constrained_VB_style_Gamma_mixture",
            "fit_note": note,
        })
    return rows, objective


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    mask_img = nib.load(str(MASK_PATH))
    mask = np.asarray(mask_img.dataobj) > 0
    labels = pd.read_csv(LABELS)
    healthy = labels.loc[labels.actual == 0, "uid"].astype(str).sort_values().tolist()

    mode_rows = []
    for i, uid in enumerate(healthy, 1):
        src = INPUT / f"{uid}.nii.gz"
        img = nib.load(str(src))
        a = np.asarray(img.dataobj, dtype=np.float32)
        if a.shape != mask.shape or not np.allclose(a.shape and img.affine, mask_img.affine, atol=1e-5):
            raise ValueError(f"{uid}: scan/template geometry mismatch")
        pos = a[mask & np.isfinite(a) & (a > 0)]
        mode, mode_bin, bin_count = histogram_mode(pos, 500)
        scaled = np.zeros_like(a, dtype=np.float32)
        inside = mask & np.isfinite(a)
        scaled[inside] = a[inside] / mode
        scaled_mode, _, _ = histogram_mode(scaled[mask & (scaled > 0)], 500)
        if not (0.98 <= scaled_mode <= 1.02):
            raise AssertionError(f"{uid}: scaled mode {scaled_mode:.6f} outside [0.98,1.02]")
        hdr = img.header.copy(); hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(scaled, img.affine, hdr), str(NORMALIZED / f"{uid}.nii.gz"))
        mode_rows.append({"uid": uid, "positive_voxels": len(pos), "mode": mode,
                          "mode_bin": mode_bin, "mode_bin_count": bin_count,
                          "scaled_mode": scaled_mode})
        if i % 50 == 0 or i == len(healthy):
            print(f"normalized={i}/{len(healthy)}", flush=True)
    pd.DataFrame(mode_rows).to_csv(OUT / "mode_scaling_audit.csv", index=False)

    rng = np.random.default_rng(SEED)
    values = {}
    samples = []
    for i, uid in enumerate(healthy):
        a = np.asarray(nib.load(str(NORMALIZED / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        v = a[mask & np.isfinite(a) & (a > 0)]
        values[uid] = v
        samples.append(rng.choice(v, min(10000, len(v)), replace=False))
    sample = np.concatenate(samples)
    lo, hi = np.percentile(np.log10(sample), [0.1, 99.9])
    edges = np.linspace(lo, hi, 128 + 1)
    hist = []
    for uid in healthy:
        counts, _ = np.histogram(np.clip(np.log10(values[uid]), lo, hi), bins=edges)
        hist.append(counts / max(counts.sum(), 1))
    features = PCA(n_components=0.99, svd_solver="full", random_state=SEED).fit_transform(np.sqrt(np.asarray(hist)))
    assignment = KMeans(n_clusters=10, n_init=50, random_state=SEED).fit_predict(features)
    assign = pd.DataFrame({"uid": healthy, "cluster": assignment})
    assign.to_csv(OUT / "healthy_cluster_assignments.csv", index=False)

    comps, summaries = [], []
    for cluster in range(10):
        uids = assign.loc[assign.cluster == cluster, "uid"].tolist()
        x = np.concatenate([values[u] for u in uids])
        rows, objective = fit_cluster(x, cluster)
        comps.extend(rows)
        summaries.append({"cluster": cluster, "healthy_scans": len(uids),
                          "positive_voxels": len(x), "objective": objective,
                          "status": "completed"})
        print(f"cluster={cluster} healthy_scans={len(uids)} voxels={len(x)}", flush=True)
    pd.DataFrame(comps).to_csv(OUT / "gamma_mixture_components.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "cluster_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "healthy_scans": len(healthy), "clusters": 10, "histogram_bins_for_mode": 500,
        "normalization": "positive voxels inside MNI152 brain mask divided by individual 500-bin histogram mode",
        "normalization_assertion": "scaled histogram mode in [0.98, 1.02] for every healthy scan",
        "component_1": "background initialized from x <= 1.3; pi1=0.93; mean constrained to [0.90,1.10]",
        "component_2": "striatal initialized from x > 1.5 or 95th percentile; pi2=0.07; mode constrained >=1.5",
        "mixture_method": "constrained VB-style Gamma responsibilities with moment updates",
        "paper_note": "Not a line-by-line implementation of the article's Laplace/Taylor shape posterior.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
