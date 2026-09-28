#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'submission_src_ensemble/assets/template.nii'
out_png = ROOT / 'results/template_top1_first3rows_green.png'
img = nib.load(str(src)); a = img.get_fdata().astype(np.float32)
pos = a[np.isfinite(a) & (a > 0)]
cutoff = float(np.percentile(pos, 99.0))
top1 = np.isfinite(a) & (a >= cutoff)
green = top1.copy(); green[3:, :, :] = False
coords = [(a.shape[2]//2,'axial'),(a.shape[1]//2,'coronal'),(a.shape[0]//2,'sagittal')]
def plane(v,i,name): return np.rot90((v[:,:,i] if name=='axial' else v[:,i,:] if name=='coronal' else v[i,:,:]).T)
vmax = np.percentile(pos, 99.5)
fig, ax = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
for j,(i,name) in enumerate(coords):
    base = plane(a,i,name); g = plane(green,i,name)
    ax[0,j].imshow(base,cmap='gray',vmin=0,vmax=vmax); ax[0,j].set_title(f'Original {name}')
    ax[1,j].imshow(base,cmap='gray',vmin=0,vmax=vmax)
    ax[1,j].imshow(np.ma.masked_where(~g,g),cmap='Greens',vmin=0,vmax=1,alpha=.95)
    ax[1,j].set_title(f'Top 1% in rows 0–2 — {name}')
    ax[0,j].axis('off'); ax[1,j].axis('off')
fig.suptitle(f'Template top-1% voxels restricted to first 3 rows | cutoff={cutoff:.6g} | green voxels={int(green.sum())}',fontsize=13)
fig.savefig(out_png,dpi=180); plt.close(fig)
print(f'cutoff={cutoff}'); print(f'top1_voxels={int(top1.sum())}'); print(f'green_voxels={int(green.sum())}'); print(out_png)
