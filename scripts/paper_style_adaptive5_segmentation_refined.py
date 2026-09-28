#!/usr/bin/env python3
"""Refined paper-style segmentation only: adaptive five slices, MIP/non-MIP."""
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, binary_opening, binary_closing, generate_binary_structure

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'; OUT=ROOT/'results/paper_style_adaptive5_segmentation_refined'; SLABS=5

def otsu(v):
 v=np.asarray(v,float); h,e=np.histogram(v,bins=256); c=(e[:-1]+e[1:])/2; p=h/max(h.sum(),1); w=np.cumsum(p); m=np.cumsum(p*c); b=(m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12); return float(c[np.argmax(b)])

def components(image,brain2d):
 pos=image[brain2d&(image>0)&np.isfinite(image)]
 if pos.size<10:return np.zeros(image.shape,bool)
 lo,hi=np.percentile(pos,[1,99]); x=np.clip((image-lo)/max(hi-lo,1e-8),0,1); x=x+1*(x-gaussian_filter(x,1)); search=brain2d.copy(); sx,sy=x.shape; search[:int(.15*sx),:]=False; search[int(.85*sx):,:]=False; search[:,:int(.15*sy)]=False; search[:,int(.85*sy):]=False; vals=x[search&(x>0)]; hot=search&(x>=max(otsu(vals),float(np.percentile(vals,70)))); st=generate_binary_structure(2,1); hot=binary_opening(hot,st,1); hot=binary_closing(hot,st,1); comp,n=label(hot,st); mid=sx/2; cand=[]
 for k in range(1,n+1):
  m=comp==k; area=int(m.sum()); q=np.argwhere(m)
  if area<3 or area>.12*search.sum():continue
  cand.append({'m':m,'area':area,'cx':float(q[:,0].mean()),'cy':float(q[:,1].mean()),'score':float(x[m].mean())*np.sqrt(area)})
 left=[q for q in cand if q['cx']<mid]; right=[q for q in cand if q['cx']>=mid]; best=None
 for l in left:
  for r in right:
   ar=min(l['area'],r['area'])/max(l['area'],r['area']); yd=abs(l['cy']-r['cy']); ld=abs(l['cx']-mid); rd=abs(r['cx']-mid); mirror=abs(ld-rd); score=np.sqrt(l['score']*r['score'])*(0.5+0.5*ar)*np.exp(-yd/(.12*sy))*np.exp(-mirror/(.12*sy))
   if best is None or score>best[0]:best=(score,l,r)
 if best is None:return np.zeros(image.shape,bool)
 return best[1]['m']|best[2]['m']

def main():
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'mip_masks').mkdir(exist_ok=True); (OUT/'nonmip_masks').mkdir(exist_ok=True); windows=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0; uids=sorted(windows.index.astype(str)); samples=[]
 for i,uid in enumerate(uids,1):
  img=nib.load(str(REG/f'{uid}.nii.gz')); a=np.asarray(img.dataobj,dtype=np.float32); start=int(windows.loc[uid,'start']); end=int(windows.loc[uid,'end']); slab=a[:,:,start:end+1]; bm=brain[:,:,start:end+1]; mip=slab.max(2); mipbrain=bm.any(2); mm=components(mip,mipbrain); nm=np.zeros(a.shape,bool)
  for z in range(start,end+1):nm[:,:,z]=components(a[:,:,z],brain[:,:,z])
  m3=np.zeros(a.shape,np.uint8); m3[:,:,start:end+1]=np.repeat(mm[:,:,None],end-start+1,2).astype(np.uint8); n3=nm.astype(np.uint8); nib.save(nib.Nifti1Image(m3,img.affine,img.header),str(OUT/'mip_masks'/f'{uid}_mip_striatum_mask.nii.gz')); nib.save(nib.Nifti1Image(n3,img.affine,img.header),str(OUT/'nonmip_masks'/f'{uid}_nonmip_striatum_mask.nii.gz'))
  if len(samples)<20:samples.append((uid,a,mm,nm,start,end))
  if i%100==0 or i==len(uids):print(f'processed={i}/{len(uids)}',flush=True)
 for kind,path in [('mip',OUT/'sample_montage_mip.png'),('nonmip',OUT/'sample_montage_nonmip.png')]:
  fig,ax=plt.subplots(4,5,figsize=(15,11),facecolor='white')
  for q,(uid,a,mm,nm,start,end) in enumerate(samples):
   base=a[:,:,start:end+1].max(2); mask=mm if kind=='mip' else nm[:,:,start:end+1].max(2); p=base[base>0]; lo,hi=np.percentile(p,[1,99]); ax.flat[q].imshow(np.clip((base-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower'); ax.flat[q].contour(mask.T,levels=[.5],colors='red',linewidths=1); ax.flat[q].set_title(f'{uid}\nz={start}-{end}',fontsize=8); ax.flat[q].set_xticks([]);ax.flat[q].set_yticks([])
  fig.suptitle(f'Refined paper-style segmentation — adaptive five slices — {kind.upper()}',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(path,dpi=150);plt.close(fig)
 pd.DataFrame({'uid':uids,'window_start':[int(windows.loc[u,'start']) for u in uids],'window_end':[int(windows.loc[u,'end']) for u in uids]}).to_csv(OUT/'manifest.csv',index=False);print(OUT)
if __name__=='__main__':main()
