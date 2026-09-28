#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
template_path = ROOT / 'submission_src_ensemble/assets/template.nii'
caudate_path = ROOT / 'assets/atlases/fsl/masks/caudate_bilateral.nii.gz'
out = ROOT / 'results/template_caudate_first3rows_green.png'
template = nib.load(str(template_path)); a = template.get_fdata().astype(np.float32)
cmask_img = nib.load(str(caudate_path)); cmask = cmask_img.get_fdata() > 0
if cmask.shape != a.shape or not np.allclose(cmask_img.affine, template.affine):
    raise ValueError('Caudate mask is not on the template grid')
occupied_rows = np.flatnonzero(cmask.any(axis=(1,2)))
first_rows = np.array([70, 71, 72])
green = np.zeros_like(cmask, dtype=bool)
green[first_rows, :, :] = cmask[first_rows, :, :]
coords = [(a.shape[2]//2,'axial'),(a.shape[1]//2,'coronal'),(a.shape[0]//2,'sagittal')]
def plane(v,i,name): return np.rot90((v[:,:,i] if name=='axial' else v[:,i,:] if name=='coronal' else v[i,:,:]).T)
pos = a[a > 0]; vmax = np.percentile(pos, 99.5)
fig, ax = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
for j,(i,name) in enumerate(coords):
    base = plane(a,i,name); g = plane(green,i,name)
    ax[0,j].imshow(base,cmap='gray',vmin=0,vmax=vmax); ax[0,j].set_title(f'Original {name}')
    ax[1,j].imshow(base,cmap='gray',vmin=0,vmax=vmax); ax[1,j].contour(g,levels=[0.5],colors='lime',linewidths=1.8); ax[1,j].set_title(f'Caudate rows {first_rows.tolist()} — {name}')
    ax[0,j].axis('off'); ax[1,j].axis('off')
fig.suptitle(f'First three occupied caudate rows green | rows={first_rows.tolist()} | voxels={int(green.sum())}',fontsize=13)
fig.savefig(out,dpi=180); plt.close(fig)
print(f'occupied_row_range={int(occupied_rows.min())}-{int(occupied_rows.max())}')
print(f'requested_rows={first_rows.tolist()}')
print(f'green_voxels={int(green.sum())}')
print(out)
