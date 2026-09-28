#!/usr/bin/env python3

import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results" / "cv5_submission_flip_clip" / "oof_predictions.csv"
VOLUMES = ROOT / "results" / "ants_rigid_brainmask_mode_normalized" / "volumes"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "healthy_10cluster_gamma_mixtures"
N_BINS = 128
RNG_SEED = 20260914


def build_healthy_clusters(uids, mask):
    rng = np.random.default_rng(RNG_SEED)
    samples = []
    for uid in uids:
        a = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        v = a[mask & np.isfinite(a) & (a > 0)]
        samples.append(rng.choice(v, 10000, replace=False) if len(v) > 10000 else v)
    sample_log = np.log10(np.concatenate(samples))
    lo, hi = np.percentile(sample_log, [0.1, 99.9])
    edges = np.linspace(float(lo), float(hi), N_BINS + 1)
    histograms = []
    for uid in uids:
        a = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
        v = a[mask & np.isfinite(a) & (a > 0)]
        counts, _ = np.histogram(np.clip(np.log10(v), lo, hi), bins=edges)
        histograms.append(counts / counts.sum())
    features = PCA(n_components=0.99, svd_solver="full").fit_transform(np.sqrt(np.asarray(histograms)))
    labels = KMeans(n_clusters=10, n_init=20, random_state=RNG_SEED).fit_predict(features)
    return labels


def gamma_logpdf(x, shape, scale):
    return (shape - 1.0) * np.log(x) - x / scale - gammaln(shape) - shape * np.log(scale)


def initialize(x, rng):
    q = np.quantile(x, [0.30, 0.75])
    means = np.maximum(q, 1e-6)
    variance = max(float(np.var(x)), 1e-6)
    shapes = np.clip(means**2 / variance, 0.15, 1e5)
    scales = np.maximum(means / shapes, 1e-6)
    weights = np.array([0.6, 0.4], dtype=float)
    if rng.random() < 0.5:
        means = means[::-1]; shapes = shapes[::-1]; scales = scales[::-1]
    return weights, shapes, scales


def fit_vb_style(x, seed):
    rng = np.random.default_rng(seed)
    weights, shapes, scales = initialize(x, rng)
    alpha_prior = np.ones(2, dtype=float)
    prior_strength = 1.0
    previous = -np.inf
    for iteration in range(150):
        expected_log_pi = digamma(alpha_prior + weights * len(x)) - digamma(alpha_prior.sum() + len(x))
        log_r = np.column_stack([
            expected_log_pi[k] + gamma_logpdf(x, shapes[k], scales[k]) for k in range(2)
        ])
        normalizer = log_r.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_r - normalizer)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        nk = responsibilities.sum(axis=0)
        alpha = alpha_prior + nk
        for k in range(2):
            weighted_sum = float(np.dot(responsibilities[:, k], x))
            weighted_sq = float(np.dot(responsibilities[:, k], x * x))
            mean = (weighted_sum + prior_strength) / (nk[k] + prior_strength)
            second = (weighted_sq + 2.0 * prior_strength) / (nk[k] + prior_strength)
            variance = max(second - mean * mean, 1e-8)
            shapes[k] = float(np.clip(mean * mean / variance, 0.05, 1e6))
            scales[k] = float(max(variance / mean, 1e-8))
        weights = alpha / alpha.sum()
        objective = float(np.sum(normalizer[:, 0] + np.log(np.exp(log_r - normalizer).sum(axis=1))))
        if not np.isfinite(objective) or not np.all(np.isfinite([*weights, *shapes, *scales])):
            raise FloatingPointError("non-finite VB-style Gamma parameters")
        if abs(objective - previous) < 1e-5 * (1.0 + abs(objective)):
            break
        previous = objective
    return weights, shapes, scales, responsibilities, iteration + 1, objective


def fit_em_fallback(x, seed):
    rng = np.random.default_rng(seed)
    weights, shapes, scales = initialize(x, rng)
    previous = -np.inf
    for iteration in range(200):
        log_r = np.column_stack([
            np.log(max(weights[k], 1e-12)) + gamma_logpdf(x, shapes[k], scales[k]) for k in range(2)
        ])
        normalizer = log_r.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_r - normalizer)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        nk = responsibilities.sum(axis=0)
        weights = nk / len(x)
        for k in range(2):
            mean = max(float(np.dot(responsibilities[:, k], x) / nk[k]), 1e-8)
            second = float(np.dot(responsibilities[:, k], x * x) / nk[k])
            variance = max(second - mean * mean, 1e-8)
            shapes[k] = float(np.clip(mean * mean / variance, 0.05, 1e6))
            scales[k] = float(max(variance / mean, 1e-8))
        objective = float(np.sum(normalizer[:, 0] + np.log(np.exp(log_r - normalizer).sum(axis=1))))
        if abs(objective - previous) < 1e-5 * (1.0 + abs(objective)):
            break
        previous = objective
    return weights, shapes, scales, responsibilities, iteration + 1, objective


def fit_cluster(x, cluster):
    # Fit on a deterministic cap only if a pathological volume count is encountered.
    # Normal clusters fit all concatenated positive voxels.
    if len(x) > 5_000_000:
        rng = np.random.default_rng(RNG_SEED + cluster)
        x = rng.choice(x, 5_000_000, replace=False)
        fit_note = "deterministic 5M voxel cap"
    else:
        fit_note = "all positive in-mask voxels"
    best = None
    for seed in (RNG_SEED + cluster, 17 + cluster, 41 + cluster):
        try:
            result = fit_vb_style(x, seed)
            if best is None or result[-1] > best[-1]:
                best = result
        except Exception:
            continue
    method = "VB_style_Gamma_mixture"
    if best is None:
        best = fit_em_fallback(x, RNG_SEED + cluster)
        method = "EM_Gamma_fallback"
    weights, shapes, scales, responsibilities, iterations, objective = best
    order = np.argsort(shapes * scales)
    responsibilities = responsibilities[:, order]
    weights, shapes, scales = weights[order], shapes[order], scales[order]
    labels = (np.arange(2) == np.argmax(responsibilities, axis=1)[:, None]).argmax(axis=1)
    components = []
    for k, name in enumerate(("background_uptake", "specific_striatal_uptake")):
        mode = max((shapes[k] - 1.0) * scales[k], 0.0)
        mean = shapes[k] * scales[k]
        components.append({
            "cluster": cluster, "component": k, "label": name,
            "mixture_weight": float(weights[k]), "shape": float(shapes[k]),
            "scale": float(scales[k]), "mean": float(mean), "mode": float(mode),
            "iterations": int(iterations), "fit_method": method,
            "fit_voxels": int(len(x)), "fit_note": fit_note,
            "assigned_voxels": int((labels == k).sum()),
        })
    return components, float(objective), method, len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)
    healthy_uids = sorted(labels.loc[labels.actual == 0, "uid"].tolist())
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0
    cluster_labels = build_healthy_clusters(healthy_uids, mask)
    assignments = pd.DataFrame({"uid": healthy_uids, "cluster": cluster_labels})
    assignments.to_csv(OUT / "healthy_10cluster_assignments.csv", index=False)

    components = []
    cluster_summary = []
    for cluster in range(10):
        uids = assignments.loc[assignments.cluster == cluster, "uid"].tolist()
        arrays = []
        for uid in uids:
            a = np.asarray(nib.load(str(VOLUMES / f"{uid}.nii.gz")).dataobj, dtype=np.float32)
            arrays.append(a[mask & np.isfinite(a) & (a > 0)])
        values = np.concatenate(arrays)
        fitted, objective, method, fit_voxels = fit_cluster(values, cluster)
        components.extend(fitted)
        cluster_summary.append({
            "cluster": cluster, "scans": len(uids), "positive_voxels": int(len(values)),
            "fit_voxels": fit_voxels, "fit_method": method, "objective": objective,
        })
        print(f"cluster={cluster} scans={len(uids)} voxels={len(values)} method={method}", flush=True)

    pd.DataFrame(components).to_csv(OUT / "gamma_mixture_components.csv", index=False)
    pd.DataFrame(cluster_summary).to_csv(OUT / "cluster_summary.csv", index=False)
    summary = {
        "healthy_scans": len(healthy_uids), "clusters": 10,
        "distribution": "two-component Gamma mixture per healthy histogram cluster",
        "normalization": "MNI brain-mask positive voxels divided by per-scan primary histogram mode",
        "primary_fit": "analytic VB-style Gamma responsibilities with Dirichlet mixing prior and moment-updated shape/scale",
        "fallback": "Gamma-mixture EM with moment updates",
        "component_order": "lower mean/mode = background_uptake; higher mean/mode = specific_striatal_uptake",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
