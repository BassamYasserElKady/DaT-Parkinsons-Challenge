from pathlib import Path
import csv
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
UID='koefukpv'
BASE=ROOT/'results/ants_registration_mni'/f'{UID}.nii.gz'
ROUTE=ROOT/'results/route_b_ants_random25_affine'/f'{UID}.nii.gz'
AUDIT=ROOT/'results/route_b_ants_random25_affine/transform_audit.csv'
OUT=ROOT/'results/route_b_ants_random25_affine'/f'{UID}_before_after.png'

def prep(a):
    a=np.asarray(a,dtype=float); a=np.maximum(a,0)
    vmax=np.percentile(a[a>0],99.5) if np.any(a>0) else 1
    return a,vmax

def main():
    before,v1=prep(nib.load(str(BASE)).get_fdata())
    after,v2=prep(nib.load(str(ROUTE)).get_fdata()); vmax=max(v1,v2)
    z=tuple(int(s//2) for s in before.shape)
    audit={r['uid']:r for r in csv.DictReader(AUDIT.open())}[UID]
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    views=[('Axial',lambda x:np.rot90(x[:,:,z[2]])),('Coronal',lambda x:np.rot90(x[:,z[1],:])),('Sagittal',lambda x:np.rot90(x[z[0],:,:]))]
    for j,(name,fn) in enumerate(views):
        ax[0,j].imshow(fn(before),cmap='gray',vmin=0,vmax=vmax); ax[0,j].set_title(f'{name} — original registered')
        ax[1,j].imshow(fn(after),cmap='gray',vmin=0,vmax=vmax); ax[1,j].set_title(f'{name} — Route B ANTs')
        ax[0,j].axis('off'); ax[1,j].axis('off')
    fig.suptitle(f"{UID} | max shear={float(audit['shear_max_abs']):.4f} | scales={audit['sx']}, {audit['sy']}, {audit['sz']}")
    fig.savefig(OUT,dpi=180); plt.close(fig)
    print(OUT)

if __name__=='__main__':main()
