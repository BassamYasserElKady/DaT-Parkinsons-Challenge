#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
img = nib.load(str(ROOT/'submission_src_ensemble/assets/template.nii'))
a = img.get_fdata().astype(np.float32)
rows = [70, 71, 72]
green = np.zeros_like(a, dtype=bool)
green[rows, :, :] = True
out = ROOT/'results/template_rows70_71_72_green.png'
coords = [(a.shape[2]//2,'axial'),(a.shape[1]//2,'coronal'),(a.shape[0]//2,'sagittal')]
def plane(v,i,name): return np.rot90((v[:,:,i] if name=='axial' else v[:,i,:] if name=='coronal' else v[i,:,:]).T)
pos = a[a>0]; vmax = np.percentile(pos,99.5)
fig, ax = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for j,(i,name) in enumerate(coords):
    base=plane(a,i,name); g=plane(green,i,name)
    ax[j].imshow(base,cmap='gray',vmin=0,vmax=vmax)
    ax[j].imshow(np.ma.masked_where(~g,g),cmap='Greens',vmin=0,vmax=1,alpha=.95)
    ax[j].set_title(f'Rows 70, 71, 72 — {name}'); ax[j].axis('off')
fig.suptitle('Template with requested voxel rows highlighted green',fontsize=13)
fig.savefig(out,dpi=180); plt.close(fig)
print(out)
