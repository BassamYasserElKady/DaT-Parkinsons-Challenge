#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import ants
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/audit_rigid_vs_affine_top20'
OUT = AUDIT / 'pngs'
UIDS = [
    'bmae38sr','7875aud9','xid4n1u3','y08p0fod','uh17j8qo',
    'bglmy62x','c1m3o55i','rqq5qob8','07gbasf5','vfua9bq1',
    '0imjyjcm','344mw04i','0uc87nxp','l5u7nted','nal3gywi',
    'nkb4ehou','6tx0tbkg','ccprhvlz','f5gzpl1q','dhvs41lq',
]

def slices(a):
    z, y, x = (n // 2 for n in a.shape)
    return [a[:, :, z].T, a[:, y, :].T, a[x, :, :].T]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for uid in UIDS:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for row, method in enumerate(('Rigid', 'Affine')):
            img = ants.image_read(str(AUDIT / 'warped' / f'{uid}_{method}.nii.gz')).numpy()
            for col, (ax, sl) in enumerate(zip(axes[row], slices(img))):
                ax.imshow(np.rot90(sl), cmap='gray')
                ax.set_title(f'{method} — {("axial", "coronal", "sagittal")[col]}')
                ax.axis('off')
        fig.suptitle(f'{uid}: rigid versus affine registration', fontsize=14)
        fig.savefig(OUT / f'{uid}_rigid_affine.png', dpi=150)
        plt.close(fig)
    zip_path = AUDIT / 'rigid_affine_registration_pngs.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')):
            z.write(p, p.name)
    print(zip_path)

if __name__ == '__main__':
    main()
