#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
scan = nib.load(str(ROOT/'results/ants_registration_mni/bmae38sr.nii.gz')).get_fdata()
mask = nib.load(str(ROOT/'assets/atlases/fsl/masks/striatal_voi_mask_LAS.nii.gz')).get_fdata() > 0
out = ROOT/'results/top20_ensemble_gradcam/bmae38sr_striatal_overlay_corrected.png'
zs = range(26, 34)
vmax = np.percentile(scan[scan > 0], 99.5)
fig, ax = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
for a, z in zip(ax.flat, zs):
    base = np.rot90(scan[:, :, z].T)
    m = np.rot90(mask[:, :, z].T)
    a.imshow(base, cmap='gray', vmin=0, vmax=vmax)
    a.contour(m, levels=[0.5], colors='red', linewidths=1.4)
    a.set_title(f'bmae38sr — axial z={z}')
    a.axis('off')
fig.suptitle('Rigid-registered bmae38sr with template striatal VOI overlay (z=26–33)', fontsize=14)
fig.savefig(out, dpi=180)
plt.close(fig)
print(out)
