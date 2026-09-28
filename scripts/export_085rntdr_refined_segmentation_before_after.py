#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt
from scipy.stats import gamma
from paper_style_adaptive5_segmentation_refined import components
import nested_gcdf_shape40_mask_cv5 as gcdf

ROOT=Path(__file__).resolve().parents[1]; UID='085rntdr'; OUT=ROOT/'results/validate_gcdf_regions'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; PARAMS=ROOT/'results/nested_gcdf_shape40_mask_cv5/fold_gamma_parameters.csv'; ASSIGN=ROOT/'results/nested_gcdf_shape40_mask_cv5/fold_gcdf_features.csv'; BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'

def norm(im):
 p=im[np.isfinite(im)&(im>0)]; lo,hi=np.percentile(p,[1,99.5]); return np.clip((im-lo)/max(hi-lo,1e-8),0,1)

def main():
 brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0; a=gcdf.scaled(UID,brain); assign=pd.read_csv(ASSIGN); ar=assign[assign.uid==UID].iloc[0]; pars=pd.read_csv(PARAMS); par=pars[(pars.fold==int(ar.fold))&(pars.cluster==int(ar.cluster))].iloc[0]; b=gamma.cdf(np.maximum(a,0),a=float(par['shape']),scale=float(par['scale'])); windows=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); start=int(windows.loc[UID,'start']); end=int(windows.loc[UID,'end']); fig,ax=plt.subplots(2,5,figsize=(17,7),facecolor='white')
 for j,z in enumerate(range(start,end+1)):
  for r,vol,title in [(0,a,'Before GCDF'),(1,b,'After GCDF')]:
   im=vol[:,:,z]; mask=components(im,brain[:,:,z]); ax[r,j].imshow(norm(im).T,cmap='gray',origin='lower',vmin=0,vmax=1); ax[r,j].contour(mask.T,levels=[.5],colors='red',linewidths=1.4); ax[r,j].set_title(f'{title}\nz={z}',fontsize=10); ax[r,j].set_xticks([]);ax[r,j].set_yticks([])
 fig.suptitle(f'{UID} — refined per-slice striatal segmentation before vs after GCDF | fold={int(ar.fold)}, cluster={int(ar.cluster)}',fontsize=13); fig.tight_layout(rect=[0,0,1,.92]); path=OUT/f'{UID}_refined_segmentation_before_after_gcdf.png'; fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig); print(path)
if __name__=='__main__': main()
