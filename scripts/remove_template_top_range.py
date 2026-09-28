#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'submission_src_ensemble/assets/template.nii'
out = ROOT / 'results/template_top_removal_range'
out.mkdir(parents=True, exist_ok=True)
img = nib.load(str(src)); a = img.get_fdata().astype(np.float32)
pos = a[np.isfinite(a) & (a > 0)]
coords = [(a.shape[2]//2,'axial'),(a.shape[1]//2,'coronal'),(a.shape[0]//2,'sagittal')]
def plane(v,i,name): return np.rot90((v[:,:,i] if name=='axial' else v[:,i,:] if name=='coronal' else v[i,:,:]).T)
vmax = np.percentile(pos, 99.5)
for k in range(1, 21):
    percent = k * 0.5
    cutoff = float(np.percentile(pos, 100.0 - percent))
    removed = np.isfinite(a) & (a >= cutoff)
    modified = a.copy(); modified[removed] = 0
    stem = f'template_without_top{percent:g}'
    nii = out / f'{stem}.nii.gz'; png = out / f'{stem}.png'
    nib.save(nib.Nifti1Image(modified, img.affine, img.header.copy()), str(nii))
    fig, ax = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for j,(i,name) in enumerate(coords):
        ax[0,j].imshow(plane(a,i,name),cmap='gray',vmin=0,vmax=vmax); ax[0,j].set_title(f'Original {name}')
        ax[1,j].imshow(plane(modified,i,name),cmap='gray',vmin=0,vmax=vmax); ax[1,j].set_title(f'Top {percent:g}% removed — {name}')
        ax[0,j].axis('off'); ax[1,j].axis('off')
    fig.suptitle(f'Template top-{percent:g}% removal | cutoff={cutoff:.6g} | removed={int(removed.sum())}',fontsize=13)
    fig.savefig(png,dpi=150); plt.close(fig)
zip_path = ROOT / 'results/template_top_removal_0.5_to_10.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()): z.write(p,p.relative_to(out))
print(f'levels=20 positive_voxels={len(pos)}')
print(zip_path)
