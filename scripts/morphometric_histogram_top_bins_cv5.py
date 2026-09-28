#!/usr/bin/env python3
"""Ablate morphometric segmentation by retaining only the brightest histogram bins."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter, label, binary_opening, binary_closing, generate_binary_structure, binary_erosion
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
WIN = ROOT / 'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'
LABELS = ROOT / 'results/cv5_submission_flip_clip/oof_predictions.csv'
OUT = ROOT / 'results/morphometric_histogram_top_bins_cv5'
N_BINS = 128
TOP_BINS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
SEED = 20260931


def otsu(v):
    h, e = np.histogram(v, bins=256)
    c = (e[:-1] + e[1:]) / 2
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    m = np.cumsum(p * c)
    b = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(c[np.argmax(b)])


def segment(im, bm):
    im = im.astype(float)
    pos = im[bm & (im > 0) & np.isfinite(im)]
    if pos.size < 10:
        return np.zeros(im.shape, bool)
    lo, hi = np.percentile(pos, [1, 99])
    x = np.clip((im - lo) / max(hi - lo, 1e-8), 0, 1)
    x = x + (x - gaussian_filter(x, 1))
    sx, sy = x.shape
    search = bm.copy()
    search[:int(.15 * sx), :] = False
    search[int(.85 * sx):, :] = False
    search[:, :int(.15 * sy)] = False
    search[:, int(.85 * sy):] = False
    v = x[search & (x > 0)]
    hot = search & (x >= max(otsu(v), float(np.percentile(v, 70))))
    st = generate_binary_structure(2, 1)
    hot = binary_opening(hot, st, 1)
    hot = binary_closing(hot, st, 1)
    comp, n = label(hot, st)
    mid = sx / 2
    candidates = []
    for k in range(1, n + 1):
        m = comp == k
        area = int(m.sum())
        q = np.argwhere(m)
        if area >= 3 and area <= .12 * search.sum():
            candidates.append({'m': m, 'area': area, 'cx': float(q[:, 0].mean()),
                               'cy': float(q[:, 1].mean()),
                               'score': float(x[m].mean()) * np.sqrt(area)})
    left = [q for q in candidates if q['cx'] < mid]
    right = [q for q in candidates if q['cx'] >= mid]
    best = None
    for l in left:
        for r in right:
            score = np.sqrt(l['score'] * r['score'])
            score *= (0.5 + 0.5 * min(l['area'], r['area']) / max(l['area'], r['area']))
            score *= np.exp(-abs(l['cy'] - r['cy']) / (.12 * sy))
            score *= np.exp(-abs(abs(l['cx'] - mid) - abs(r['cx'] - mid)) / (.12 * sy))
            if best is None or score > best[0]:
                best = (score, l['m'] | r['m'])
    if best:
        return best[1]
    crossing = [q for q in candidates if abs(q['cx'] - mid) <= 2]
    if crossing:
        m = max(crossing, key=lambda z: z['score'])['m']
        l, r = m.copy(), m.copy()
        l[int(mid):, :] = False
        r[:int(mid), :] = False
        if l.sum() >= 3 and r.sum() >= 3:
            return l | r
    return np.zeros(im.shape, bool)


def shape(mask, spacing=2.):
    yy, xx = np.where(mask)
    if len(xx) < 3:
        return [0.] * 6
    area = len(xx) * spacing * spacing
    per = (mask & ~binary_erosion(mask)).sum() * spacing
    circ = 4 * np.pi * area / max(per * per, 1e-12)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.column_stack([xx, yy]) * spacing, rowvar=False)))[::-1]
    major = 4 * np.sqrt(max(ev[0], 0))
    minor = 4 * np.sqrt(max(ev[-1], 0))
    ecc = np.sqrt(max(0, 1 - (minor / max(major, 1e-12)) ** 2))
    return [area, per, circ, major, minor, ecc]


def retain_top_bins(raw, bm, n_top):
    pos = raw[bm & (raw > 0) & np.isfinite(raw)]
    if pos.size == 0:
        return np.zeros_like(raw)
    lo, hi = float(pos.min()), float(pos.max())
    if hi <= lo or n_top >= N_BINS:
        return raw.copy()
    edges = np.linspace(lo, hi, N_BINS + 1)
    cutoff = edges[max(0, N_BINS - n_top)]
    return np.where(raw >= cutoff, raw, 0.0)


def main():
    labels = pd.read_csv(LABELS, dtype={'uid': str}).sort_values('uid').reset_index(drop=True)
    windows = pd.read_csv(WIN, dtype={'uid': str}).set_index('uid')
    brain = np.asarray(nib.load(str(ROOT / 'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz')).dataobj) > 0
    valid = []
    for u in labels.uid.astype(str):
        if u in windows.index and (REG / f'{u}.nii.gz').exists():
            valid.append(u)
    labels = labels[labels.uid.isin(valid)].sort_values('uid').reset_index(drop=True)
    y = labels.actual.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    all_results, fold_results = [], []
    cached = {}
    for u in labels.uid.astype(str):
        a = np.asarray(nib.load(str(REG / f'{u}.nii.gz')).dataobj, dtype=np.float32)
        s, e = int(windows.loc[u, 'start']), int(windows.loc[u, 'end'])
        slab = a[:, :, s:e + 1]
        cached[u] = (slab.max(2), brain[:, :, s:e + 1].any(2))
    for n_top in TOP_BINS:
        rows = []
        for u in labels.uid.astype(str):
            raw, bm = cached[u]
            filtered = retain_top_bins(raw, bm, n_top)
            m = segment(filtered, bm)
            grid = np.indices(m.shape)
            left = m & (grid[0] >= 45)
            right = m & (grid[0] < 45)
            rows.append(shape(left) + shape(right))
        X = np.asarray(rows, float)
        p = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(cv.split(X, y), 1):
            model = make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=SEED + fold))
            model.fit(X[tr], y[tr])
            p[va] = model.predict_proba(X[va])[:, 1]
            pred = (p[va] >= .5).astype(int)
            fold_results.append({'top_bins': n_top, 'fold': fold,
                                 'auroc': roc_auc_score(y[va], p[va]),
                                 'log_loss': log_loss(y[va], p[va], labels=[0, 1]),
                                 'accuracy': accuracy_score(y[va], pred),
                                 'incorrect': int(np.sum(pred != y[va]))})
        pred = (p >= .5).astype(int)
        result = {'top_bins': n_top, 'n': len(y), 'auroc': roc_auc_score(y, p),
                  'log_loss': log_loss(y, p, labels=[0, 1]),
                  'accuracy': accuracy_score(y, pred),
                  'incorrect': int(np.sum(pred != y))}
        all_results.append(result)
        print(result, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(fold_results).to_csv(OUT / 'fold_metrics.csv', index=False)
    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == '__main__':
    main()
