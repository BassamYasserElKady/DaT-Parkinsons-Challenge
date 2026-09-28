#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'submission_src_ensemble/assets/template.nii'
out_nii = ROOT / 'results/template_without_top025.nii.gz'
out_png = ROOT / 'results/template_without_top025.png'

img = nib.load(str(src)); a = img.get_fdata().astype(np.float32)
pos = a[np.isfinite(a) & (a > 0)]
cutoff = float(np.percentile(pos, 99.75))
removed = np.isfinite(a) & (a >= cutoff)
modified = a.copy(); modified[removed] = 0
out_nii.parent.mkdir(parents=True, exist_ok=True)
nib.save(nib.Nifti1Image(modified, img.affine, img.header.copy()), str(out_nii))

fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
coords = [(a.shape[2]//2, 'axial'), (a.shape[1]//2, 'coronal'), (a.shape[0]//2, 'sagittal')]
def plane(v, i, name):
    if name == 'axial': return np.rot90(v[:, :, i].T)
    if name == 'coronal': return np.rot90(v[:, i, :].T)
    return np.rot90(v[i, :, :].T)
vmax = np.percentile(pos, 99.5)
for j, (i, name) in enumerate(coords):
    ax[0,j].imshow(plane(a, i, name), cmap='gray', vmin=0, vmax=vmax); ax[0,j].set_title(f'Original {name}')
    ax[1,j].imshow(plane(modified, i, name), cmap='gray', vmin=0, vmax=vmax); ax[1,j].set_title(f'Top 0.25% removed — {name}')
    ax[0,j].axis('off'); ax[1,j].axis('off')
fig.suptitle(f'MNI template top-0.25% removal | cutoff={cutoff:.4g} | removed voxels={int(removed.sum())}', fontsize=13)
fig.savefig(out_png, dpi=180)
plt.close(fig)
print(f'cutoff={cutoff}')
print(f'removed_voxels={int(removed.sum())}')
print(out_nii)
print(out_png)
