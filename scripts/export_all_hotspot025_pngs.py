#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
PRED = ROOT / 'results/cv5_real8_post_registration_gaussian/oof_predictions.csv'
OUT = ROOT / 'results/all_hotspot025_pngs'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED).set_index('uid')
    paths = sorted(REG.glob('*.nii.gz'))
    for n, path in enumerate(paths, 1):
        uid = path.name.removesuffix('.nii.gz')
        if uid not in pred.index: continue
        scan = nib.load(str(path)).get_fdata()
        positive = scan[np.isfinite(scan) & (scan > 0)]
        if positive.size == 0: continue
        cutoff = float(np.percentile(positive, 99.75))
        hot = np.isfinite(scan) & (scan >= cutoff)
        active = np.flatnonzero(hot.sum(axis=(0, 1)))
        if len(active):
            zs = list(np.unique(np.linspace(active.min(), active.max(), 8).round().astype(int)))
            while len(zs) < 8: zs.append(zs[-1])
            zs = zs[:8]
        else: zs = list(range(8))
        vmax = np.percentile(positive, 99.5)
        fig, ax = plt.subplots(2, 8, figsize=(16, 5), constrained_layout=True)
        for j, z in enumerate(zs):
            base = np.rot90(scan[:, :, z].T); hm = np.rot90(hot[:, :, z].T)
            ax[0,j].imshow(base, cmap='gray', vmin=0, vmax=vmax); ax[0,j].contour(hm, levels=[0.5], colors='red', linewidths=.8); ax[0,j].set_title(f'z={z}', fontsize=8)
            ax[1,j].imshow(hm, cmap='gray', vmin=0, vmax=1); ax[1,j].axis('off'); ax[0,j].axis('off')
        y = int(pred.loc[uid, 'actual']); p = float(pred.loc[uid, 'equal_ensemble_clipped'])
        ll = -(y*np.log(np.clip(p,1e-15,1-1e-15)) + (1-y)*np.log(np.clip(1-p,1e-15,1-1e-15)))
        fig.suptitle(f'{uid} — top 0.25% voxels | cutoff={cutoff:.4g} | count={int(hot.sum())} | log loss={ll:.3f}', fontsize=10)
        fig.savefig(OUT / f'{uid}_top025.png', dpi=100); plt.close(fig)
        if n % 100 == 0 or n == len(paths): print(f'processed={n}/{len(paths)}', flush=True)
    zp = ROOT / 'results/all_hotspot025_pngs.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p, p.name)
    print(f'pngs={len(list(OUT.glob("*.png")))}')
    print(zp)

if __name__ == '__main__': main()
