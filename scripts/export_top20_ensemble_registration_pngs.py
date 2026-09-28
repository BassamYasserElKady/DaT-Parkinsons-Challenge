#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ROOT / 'results/ants_registration_mni'
SELECTIONS = ROOT / 'results/cv5_real8_post_registration_gaussian/slice_selection.csv'
OUT = ROOT / 'results/top20_ensemble_rigid_pngs'
UIDS = ['bmae38sr','7875aud9','xid4n1u3','y08p0fod','uh17j8qo','bglmy62x','c1m3o55i','rqq5qob8','07gbasf5','vfua9bq1','0imjyjcm','344mw04i','0uc87nxp','l5u7nted','nal3gywi','nkb4ehou','6tx0tbkg','ccprhvlz','f5gzpl1q','dhvs41lq']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(SELECTIONS).set_index('uid')
    for uid in UIDS:
        img = nib.load(str(REGISTERED / f'{uid}.nii.gz')).get_fdata()
        row = selections.loc[uid]
        start, end = int(row['start']), int(row['end'])
        axial = [img[:, :, z].T for z in range(start, end + 1)]
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
        vmax = np.percentile(img[img > 0], 99.5) if np.any(img > 0) else 1.0
        for ax, sl, z in zip(axes.flat, axial, range(start, end + 1)):
            ax.imshow(np.rot90(sl), cmap='gray', vmin=0, vmax=vmax)
            ax.set_title(f'Axial z={z}')
            ax.axis('off')
        axes.flat[-1].axis('off')
        fig.suptitle(f'{uid} — ensemble rigid-registered input; selected slices {start}–{end}', fontsize=14)
        fig.savefig(OUT / f'{uid}_ensemble_rigid.png', dpi=160)
        plt.close(fig)
    zip_path = ROOT / 'results/top20_ensemble_rigid_pngs.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')):
            z.write(p, p.name)
    print(zip_path)

if __name__ == '__main__':
    main()
