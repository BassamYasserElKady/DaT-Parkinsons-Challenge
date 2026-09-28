#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
PRED = ROOT / 'results/cv5_real8_post_registration_gaussian/oof_predictions.csv'
OUT = ROOT / 'results/top20_hotspot025'
UIDS = ['bmae38sr','7875aud9','xid4n1u3','y08p0fod','uh17j8qo','bglmy62x','c1m3o55i','rqq5qob8','07gbasf5','vfua9bq1','0imjyjcm','344mw04i','0uc87nxp','l5u7nted','nal3gywi','nkb4ehou','6tx0tbkg','ccprhvlz','f5gzpl1q','dhvs41lq']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED).set_index('uid')
    for uid in UIDS:
        scan = nib.load(str(REG / f'{uid}.nii.gz')).get_fdata()
        positive = scan[np.isfinite(scan) & (scan > 0)]
        cutoff = float(np.percentile(positive, 99.75))
        hot = np.isfinite(scan) & (scan >= cutoff)
        zcounts = hot.sum(axis=(0, 1)); active = np.flatnonzero(zcounts)
        if len(active):
            zs = np.unique(np.linspace(active.min(), active.max(), 8).round().astype(int))
            zs = list(zs)[:8]
            while len(zs) < 8: zs.append(zs[-1])
        else: zs = list(range(8))
        vmax = np.percentile(positive, 99.5) if positive.size else 1.0
        fig, ax = plt.subplots(2, 8, figsize=(20, 6), constrained_layout=True)
        for j, z in enumerate(zs):
            base = np.rot90(scan[:, :, z].T); hm = np.rot90(hot[:, :, z].T)
            ax[0,j].imshow(base, cmap='gray', vmin=0, vmax=vmax); ax[0,j].contour(hm, levels=[0.5], colors='red', linewidths=1.2); ax[0,j].set_title(f'z={z}')
            ax[1,j].imshow(hm, cmap='gray', vmin=0, vmax=1); ax[1,j].set_title('top 0.25%')
            ax[0,j].axis('off'); ax[1,j].axis('off')
        ll = -(int(pred.loc[uid,'actual'])*np.log(np.clip(pred.loc[uid,'equal_ensemble_clipped'],1e-15,1-1e-15)) + (1-int(pred.loc[uid,'actual']))*np.log(np.clip(1-pred.loc[uid,'equal_ensemble_clipped'],1e-15,1-1e-15)))
        fig.suptitle(f'{uid} — top 0.25% positive voxels | cutoff={cutoff:.4g}, voxels={int(hot.sum())}, log loss={ll:.3f}', fontsize=14)
        fig.savefig(OUT / f'{uid}_top025.png', dpi=160); plt.close(fig)
    zp = ROOT / 'results/top20_hotspot025.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p, p.name)
    print(zp)

if __name__ == '__main__': main()
