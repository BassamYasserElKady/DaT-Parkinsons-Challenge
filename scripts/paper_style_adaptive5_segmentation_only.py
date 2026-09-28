#!/usr/bin/env python3
"""Paper-style striatal segmentation on our adaptive five-slice windows only."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, binary_opening, binary_closing, generate_binary_structure

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; OUT=ROOT/'results/paper_style_adaptive5_segmentation_only'; N=5

def otsu(v):
 v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
 if v.size==0 or np.all(v==v[0]): return float(v[0]) if v.size else 0.
 h,e=np.histogram(v,bins=256); c=(e[:-1]+e[1:])/2; p=h/max(h.sum(),1); w=np.cumsum(p); m=np.cumsum(p*c); between=(m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12); return float(c[np.argmax(between)])

def segment2d(image):
 pos=image[np.isfinite(image)&(image>0)]
 if pos.size==0: return np.zeros(image.shape,bool)
 lo,hi=np.percentile(pos,[1,99]); x=np.clip((image-lo)/max(hi-lo,1e-8),0,1); x=x+1.*(x-gaussian_filter(x,1.0)); valid=x[np.isfinite(x)&(x>0)]; hot=x>=otsu(valid); st=generate_binary_structure(2,1); hot=binary_opening(hot,st,1); hot=binary_closing(hot,st,1); comp,n=label(hot,st); mid=hot.shape[0]/2; chosen=np.zeros_like(hot)
 for side in (comp>0)&(np.indices(comp.shape)[0]<mid),(comp>0)&(np.indices(comp.shape)[0]>=mid):
  ids=np.unique(comp[side]); ids=ids[ids>0]; best=None
  for k in ids:
   m=comp==k; area=int(m.sum()); score=float(x[m].mean())*np.sqrt(area)
   if area>=3 and (best is None or score>best[0]): best=(score,m)
  if best is not None: chosen|=best[1]
 return chosen

def main():
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'mip_masks').mkdir(exist_ok=True); (OUT/'nonmip_masks').mkdir(exist_ok=True); windows=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); uids=sorted(windows.index.astype(str)); samples=[]
 for i,uid in enumerate(uids,1):
  img=nib.load(str(REG/f'{uid}.nii.gz')); a=np.asarray(img.dataobj,dtype=np.float32); start=int(windows.loc[uid,'start']); end=int(windows.loc[uid,'end']); slab=a[:,:,start:end+1]; mip=slab.max(2); mipmask=segment2d(mip); non=np.zeros(a.shape,bool)
  for z in range(start,end+1): non[:,:,z]=segment2d(a[:,:,z])
  m3=np.zeros(a.shape,np.uint8); m3[:,:,start:end+1]=np.repeat(mipmask[:,:,None],end-start+1,2).astype(np.uint8); n3=non.astype(np.uint8)
  nib.save(nib.Nifti1Image(m3,img.affine,img.header),str(OUT/'mip_masks'/f'{uid}_mip_striatum_mask.nii.gz')); nib.save(nib.Nifti1Image(n3,img.affine,img.header),str(OUT/'nonmip_masks'/f'{uid}_nonmip_striatum_mask.nii.gz'))
  if len(samples)<20: samples.append((uid,a,mipmask,non,start,end))
  if i%100==0 or i==len(uids): print(f'processed={i}/{len(uids)}',flush=True)
 def montage(kind,path):
  fig,ax=plt.subplots(4,5,figsize=(15,11),facecolor='white')
  for q,(uid,a,mm,nm,start,end) in enumerate(samples):
   base=a[:,:,start:end+1].max(2); mask=mm if kind=='mip' else nm[:,:,start:end+1].max(2); lo,hi=np.percentile(base[base>0],[1,99]); ax.flat[q].imshow(np.clip((base-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); ax.flat[q].contour(mask.T,levels=[.5],colors='red',linewidths=1); ax.flat[q].set_title(f'{uid}\nz={start}-{end}',fontsize=8); ax.flat[q].set_xticks([]); ax.flat[q].set_yticks([])
  fig.suptitle(f'Paper-style segmentation only — adaptive five slices — {kind.upper()}',fontsize=14); fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(path,dpi=150); plt.close(fig)
 montage('mip',OUT/'sample_montage_mip.png'); montage('nonmip',OUT/'sample_montage_nonmip.png'); pd.DataFrame({'uid':uids,'window_start':[int(windows.loc[u,'start']) for u in uids],'window_end':[int(windows.loc[u,'end']) for u in uids]}).to_csv(OUT/'manifest.csv',index=False); print(OUT)
if __name__=='__main__': main()
