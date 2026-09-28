#!/usr/bin/env python3
"""Refined segmentation with midline splitting for merged bilateral hotspots."""
from pathlib import Path
import numpy as np,pandas as pd,nibabel as nib,matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,label,binary_opening,binary_closing,generate_binary_structure
ROOT=Path(__file__).resolve().parents[1];REG=ROOT/'results/ants_registration_mni';WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv';BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz';OUT=ROOT/'results/paper_style_adaptive5_segmentation_split_midline'
def otsu(v):
 h,e=np.histogram(v,bins=256);c=(e[:-1]+e[1:])/2;p=h/max(h.sum(),1);w=np.cumsum(p);m=np.cumsum(p*c);b=(m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12);return float(c[np.argmax(b)])
def seg(im,bm):
 pos=im[bm&(im>0)&np.isfinite(im)]
 if pos.size<10:return np.zeros(im.shape,bool), 'empty'
 lo,hi=np.percentile(pos,[1,99]);x=np.clip((im-lo)/max(hi-lo,1e-8),0,1);x=x+x-gaussian_filter(x,1);sx,sy=x.shape;search=bm.copy();search[:int(.15*sx),:]=False;search[int(.85*sx):,:]=False;search[:,:int(.15*sy)]=False;search[:,int(.85*sy):]=False;v=x[search&(x>0)];hot=search&(x>=max(otsu(v),float(np.percentile(v,70))));st=generate_binary_structure(2,1);hot=binary_opening(hot,st,1);hot=binary_closing(hot,st,1);comp,n=label(hot,st);mid=sx/2;c=[]
 for k in range(1,n+1):
  m=comp==k;area=int(m.sum());q=np.argwhere(m)
  if area>=3 and area<=.12*search.sum():c.append({'m':m,'area':area,'cx':float(q[:,0].mean()),'cy':float(q[:,1].mean()),'score':float(x[m].mean())*np.sqrt(area)})
 left=[q for q in c if q['cx']<mid];right=[q for q in c if q['cx']>=mid];best=None
 for l in left:
  for r in right:
   score=np.sqrt(l['score']*r['score'])*(0.5+0.5*min(l['area'],r['area'])/max(l['area'],r['area']))*np.exp(-abs(l['cy']-r['cy'])/(.12*sy))*np.exp(-abs(abs(l['cx']-mid)-abs(r['cx']-mid))/(.12*sy))
   if best is None or score>best[0]:best=(score,l['m']|r['m'])
 if best is not None:return best[1],'paired'
 crossing=[q for q in c if abs(q['cx']-mid)<=2]
 if crossing:
  q=max(crossing,key=lambda z:z['score']);m=q['m'];left=m.copy();right=m.copy();left[int(mid):,:]=False;right[:int(mid),:]=False
  if left.sum()>=3 and right.sum()>=3:return left|right,'midline_split'
 return np.zeros(im.shape,bool),'rejected'
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'mip_masks').mkdir(exist_ok=True);(OUT/'nonmip_masks').mkdir(exist_ok=True);w=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid');brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0;uids=sorted(w.index.astype(str));status=[];samples=[]
 for i,u in enumerate(uids,1):
  img=nib.load(str(REG/f'{u}.nii.gz'));a=np.asarray(img.dataobj,dtype=np.float32);s,e=int(w.loc[u,'start']),int(w.loc[u,'end']);mip=a[:,:,s:e+1].max(2);mm,st=seg(mip,brain[:,:,s:e+1].any(2));nm=np.zeros(a.shape,bool);nstat=[]
  for z in range(s,e+1):nm[:,:,z],ss=seg(a[:,:,z],brain[:,:,z]);nstat.append(ss)
  m3=np.zeros(a.shape,np.uint8);m3[:,:,s:e+1]=np.repeat(mm[:,:,None],e-s+1,2).astype(np.uint8);n3=nm.astype(np.uint8);nib.save(nib.Nifti1Image(m3,img.affine,img.header),str(OUT/'mip_masks'/f'{u}_mip_striatum_mask.nii.gz'));nib.save(nib.Nifti1Image(n3,img.affine,img.header),str(OUT/'nonmip_masks'/f'{u}_nonmip_striatum_mask.nii.gz'));status.append({'uid':u,'mip_status':st,'mip_voxels_2d':int(mm.sum()),'nonmip_voxels_3d':int(nm.sum())});
  if len(samples)<20:samples.append((u,a,mm,s,e))
  if i%100==0 or i==len(uids):print(f'processed={i}/{len(uids)}',flush=True)
 fig,ax=plt.subplots(4,5,figsize=(15,11),facecolor='white')
 for q,(u,a,m,s,e) in enumerate(samples):
  x=a[:,:,s:e+1].max(2);p=x[x>0];lo,hi=np.percentile(p,[1,99]);ax.flat[q].imshow(np.clip((x-lo)/max(hi-lo,1e-8),0,1).T,cmap='gray',origin='lower');ax.flat[q].contour(m.T,levels=[.5],colors='red');ax.flat[q].set_title(f'{u}\nz={s}-{e}',fontsize=8);ax.flat[q].set_xticks([]);ax.flat[q].set_yticks([])
 fig.suptitle('Refined segmentation with midline-split fallback — adaptive five-slice MIP',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'sample_montage_mip.png',dpi=150);plt.close(fig);pd.DataFrame(status).to_csv(OUT/'segmentation_status.csv',index=False);print(pd.Series([q['mip_status'] for q in status]).value_counts().to_string());print(OUT)
if __name__=='__main__':main()
