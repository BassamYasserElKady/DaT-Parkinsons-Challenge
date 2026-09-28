#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
OUT = ROOT / 'results/all_hotspot025_masks'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = sorted(REG.glob('*.nii.gz'))
    for n, path in enumerate(paths, 1):
        uid = path.name.removesuffix('.nii.gz')
        img = nib.load(str(path)); a = img.get_fdata()
        pos = a[np.isfinite(a) & (a > 0)]
        if pos.size == 0: continue
        cutoff = float(np.percentile(pos, 99.75))
        mask = (np.isfinite(a) & (a >= cutoff)).astype(np.uint8)
        out_img = nib.Nifti1Image(mask, img.affine, img.header.copy())
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, str(OUT / f'{uid}_top025_mask.nii.gz'))
        rows.append({'uid': uid, 'positive_voxels': int(pos.size), 'cutoff_99_75': cutoff, 'retained_voxels': int(mask.sum()), 'retained_percent_positive': float(100*mask.sum()/pos.size)})
        if n % 200 == 0 or n == len(paths): print(f'processed={n}/{len(paths)}', flush=True)
    with (OUT/'summary.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'extracted={len(rows)}')
    print(f'output={OUT}')

if __name__ == '__main__': main()
