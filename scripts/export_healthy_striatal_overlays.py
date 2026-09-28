#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
MASK = nib.load(str(ROOT/'assets/atlases/fsl/masks/striatal_voi_mask_LAS.nii.gz')).get_fdata() > 0
ROWS = ROOT / 'results/ants_registered_consecutive_resnet18_cbam/split.csv'
OUT = ROOT / 'results/healthy20_striatal_overlays'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROWS)
    uids = sorted(df.loc[df['is_pathologic'] == 0, 'uid'].astype(str).tolist())[:20]
    nz = np.argwhere(MASK)
    z0, z1 = int(nz[:, 2].min()), int(nz[:, 2].max())
    zs = list(range(z0, min(z0 + 8, z1 + 1)))
    for uid in uids:
        scan = nib.load(str(REG / f'{uid}.nii.gz')).get_fdata()
        positive = scan[scan > 0]
        vmax = np.percentile(positive, 99.5) if positive.size else 1.0
        fig, ax = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        for a, z in zip(ax.flat, zs):
            a.imshow(np.rot90(scan[:, :, z].T), cmap='gray', vmin=0, vmax=vmax)
            a.contour(np.rot90(MASK[:, :, z].T), levels=[0.5], colors='red', linewidths=1.4)
            a.set_title(f'{uid} — axial z={z}')
            a.axis('off')
        fig.suptitle(f'{uid} — healthy scan with template striatal VOI overlay', fontsize=14)
        fig.savefig(OUT / f'{uid}_striatal_overlay.png', dpi=180)
        plt.close(fig)
    zp = ROOT / 'results/healthy20_striatal_overlays.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p, p.name)
    print(f'healthy_scans={len(uids)} slices={z0}-{zs[-1]}')
    print(zp)

if __name__ == '__main__': main()
