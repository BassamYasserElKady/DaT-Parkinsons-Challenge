#!/usr/bin/env python3
"""Refit healthy Gamma mixtures with wider tail mass and striatal shape/mode floors."""

from pathlib import Path
import json
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "healthy_mode500_tail_gamma_clusters_10"
VOLUMES = BASE / "normalized_healthy_volumes"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUT = ROOT / "results" / "healthy_mode500_shape2_mode115_gamma_clusters_10"
SEED = 20260914


def logpdf(x, shape, scale):
    return (shape - 1) * np.log(x) - x / scale - gammaln(shape) - shape * np.log(scale)


def moments(x, weights, min_shape=1.05):
    n = max(float(weights.sum()), 1e-8)
    mean = max(float(np.dot(weights, x) / n), 1e-6)
    second = max(float(np.dot(weights, x * x) / n), mean * mean + 1e-8)
    var = max(second - mean * mean, 1e-8)
    shape = float(np.clip(mean * mean / var, min_shape, 1e5))
    return shape, max(mean / shape, 1e-8)


def fit(x, cluster):
    note = "all normalized voxels"
    if len(x) > 5_000_000:
        x = np.random.default_rng(SEED + cluster).choice(x, 5_000_000, replace=False)
        note = "deterministic 5M voxel cap"
    bg = x <= 1.25
    tail = x > 1.50
    if bg.sum() < 100: bg = x <= np.quantile(x, .9)
    if tail.sum() < 100: tail = x >= np.quantile(x, .98)
    s1, c1 = moments(x, bg.astype(float))
    s2, c2 = moments(x, tail.astype(float), min_shape=2.0)
    w2 = .075; w1 = .925
    mean1 = float(np.clip(s1 * c1, .90, 1.10)); c1 = mean1 / s1
    c2 = max(c2, 1.15 / (s2 - 1.0), (mean1 + .5) / s2)
    prev = -np.inf
    for it in range(250):
        lr = np.column_stack([np.log(w1) + logpdf(x, s1, c1), np.log(w2) + logpdf(x, s2, c2)])
        mx = lr.max(axis=1, keepdims=True)
        r = np.exp(lr - mx); r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w2 = float(np.clip(nk[1] / len(x), .04, .12)); w1 = 1 - w2
        s1, c1 = moments(x, r[:, 0]); mean1 = float(np.clip(s1 * c1, .90, 1.10)); c1 = mean1 / s1
        s2, c2 = moments(x, r[:, 1], min_shape=2.0)
        c2 = max(c2, 1.15 / (s2 - 1.0), (mean1 + .5) / s2)
        obj = float(np.sum(mx[:, 0] + np.log(np.exp(lr - mx).sum(axis=1))))
        if abs(obj - prev) < 1e-5 * (1 + abs(obj)): break
        prev = obj
    lab = r.argmax(axis=1)
    out=[]
    for k, (name, weight, shape, scale) in enumerate([
        ('background_uptake', w1, s1, c1), ('specific_striatal_uptake', w2, s2, c2)]):
        out.append({'cluster':cluster,'component':k,'label':name,'mixture_weight':weight,
                    'shape':shape,'scale':scale,'mean':shape*scale,
                    'mode':max((shape-1)*scale,0),'assigned_voxels':int((lab==k).sum()),
                    'fit_voxels':len(x),'iterations':it+1,
                    'fit_method':'shape2_mode115_constrained_Gamma_mixture','fit_note':note})
    return out, obj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(nib.load(str(MASK_PATH)).dataobj) > 0
    assignments = pd.read_csv(BASE / 'healthy_cluster_assignments.csv')
    values={}
    for uid in assignments.uid.astype(str):
        a=np.asarray(nib.load(str(VOLUMES/f'{uid}.nii.gz')).dataobj,dtype=np.float32)
        values[uid]=a[mask & np.isfinite(a) & (a>0)]
    rows=[]; summaries=[]
    for cluster in range(10):
        uids=assignments.loc[assignments.cluster==cluster,'uid'].astype(str).tolist()
        x=np.concatenate([values[u] for u in uids]); fitted,obj=fit(x,cluster); rows.extend(fitted)
        summaries.append({'cluster':cluster,'healthy_scans':len(uids),'positive_voxels':len(x),'objective':obj,'status':'completed'})
        print(f'cluster={cluster} healthy_scans={len(uids)} voxels={len(x)}',flush=True)
    assignments.to_csv(OUT/'healthy_cluster_assignments.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'gamma_mixture_components.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'cluster_summary.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps({'healthy_scans':len(assignments),'clusters':10,
        'pi2_bounds':[.04,.12],'pi2_initial':.075,'shape2_floor':2.0,'mode2_floor':1.15,
        'mean_constraint':'mean2 >= max(1.5, mean1 + 0.5)'},indent=2)+'\n')

if __name__ == '__main__': main()
