#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
template = nib.load(str(ROOT/'submission_src_ensemble/assets/template.nii'))
a = template.get_fdata().astype(np.float32)
caudate = nib.load(str(ROOT/'assets/atlases/fsl/masks/caudate_bilateral.nii.gz')).get_fdata() > 0
caudate_values = np.where(caudate, a, 0)
coords = [(a.shape[2]//2,'axial'),(a.shape[1]//2,'coronal'),(a.shape[0]//2,'sagittal')]
def plane(v,i,name): return np.rot90((v[:,:,i] if name=='axial' else v[:,i,:] if name=='coronal' else v[i,:,:]).T)
pos = a[a>0]; vmax = np.percentile(pos,99.5)
out = ROOT/'results/template_caudate_voxels_only.png'
fig, ax = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
for j,(i,name) in enumerate(coords):
    ax[0,j].imshow(plane(a,i,name),cmap='gray',vmin=0,vmax=vmax); ax[0,j].set_title(f'Original template — {name}')
    ax[1,j].imshow(plane(caudate_values,i,name),cmap='gray',vmin=0,vmax=vmax); ax[1,j].set_title(f'Caudate voxels only — {name}')
    ax[0,j].axis('off'); ax[1,j].axis('off')
fig.suptitle('Template caudate intensities shown without a mask overlay',fontsize=13)
fig.savefig(out,dpi=180); plt.close(fig)
print(f'caudate_voxels={int(caudate.sum())}')
print(out)
