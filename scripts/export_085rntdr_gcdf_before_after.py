#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt
from scipy.stats import gamma
import nested_gcdf_shape40_mask_cv5 as gcdf

ROOT=Path(__file__).resolve().parents[1]; UID='085rntdr'; OUT=ROOT/'results/validate_gcdf_regions'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; PARAMS=ROOT/'results/nested_gcdf_shape40_mask_cv5/fold_gamma_parameters.csv'; ASSIGN=ROOT/'results/nested_gcdf_shape40_mask_cv5/fold_gcdf_features.csv'; BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'

def main():
 brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0; a=gcdf.scaled(UID,brain); assignment=pd.read_csv(ASSIGN); row=assignment[assignment.uid==UID].iloc[0]; pars=pd.read_csv(PARAMS); par=pars[(pars.fold==int(row.fold))&(pars.cluster==int(row.cluster))].iloc[0]; windows=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); start=int(windows.loc[UID,'start']); end=int(windows.loc[UID,'end']); b=gamma.cdf(np.maximum(a,0),a=float(par['shape']),scale=float(par['scale'])); raw=a[:,:,start:end+1]; transformed=b[:,:,start:end+1]; fig,ax=plt.subplots(2,5,figsize=(17,7),facecolor='white')
 for j,z in enumerate(range(start,end+1)):
  im=raw[:,:,j]; pos=im[im>0]; lo,hi=np.percentile(pos,[1,99.5]); ax[0,j].imshow(np.clip((im-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); ax[0,j].set_title(f'Before z={z}',fontsize=10); ax[1,j].imshow(transformed[:,:,j].T,cmap='viridis',origin='lower',vmin=0,vmax=1); ax[1,j].set_title(f'After GCDF z={z}',fontsize=10)
  ax[0,j].set_xticks([]);ax[0,j].set_yticks([]);ax[1,j].set_xticks([]);ax[1,j].set_yticks([])
 fig.suptitle(f'{UID} — voxel-intensity remapping | fold={int(row.fold)}, cluster={int(row.cluster)} | shape={float(par["shape"]):.4f}, scale={float(par["scale"]):.4f}',fontsize=13); fig.tight_layout(rect=[0,0,1,.92]); path=OUT/f'{UID}_before_after_gcdf.png'; fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig); print(path)
if __name__=='__main__': main()
