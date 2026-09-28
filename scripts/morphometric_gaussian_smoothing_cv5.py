#!/usr/bin/env python3
"""Test pre-segmentation Gaussian smoothing levels with Gaussian SVM."""
from pathlib import Path
import numpy as np,pandas as pd,nibabel as nib
from scipy.ndimage import gaussian_filter, label, binary_opening, binary_closing, generate_binary_structure, binary_erosion
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
ROOT=Path(__file__).resolve().parents[1];REG=ROOT/'results/ants_registration_mni';WIN=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv';BASEMASK=ROOT/'results/paper_style_adaptive5_segmentation_refined/mip_masks';LABELS=ROOT/'results/cv5_submission_flip_clip/oof_predictions.csv';OUT=ROOT/'results/morphometric_gaussian_smoothing_cv5';SIGMAS=(.1,.25,.5,.75);SEED=20260931
def otsu(v):
 h,e=np.histogram(v,bins=256);c=(e[:-1]+e[1:])/2;p=h/max(h.sum(),1);w=np.cumsum(p);m=np.cumsum(p*c);b=(m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12);return float(c[np.argmax(b)])
def segment(im,bm,sigma):
 im=gaussian_filter(im.astype(float),sigma) if sigma>0 else im.astype(float);pos=im[bm&(im>0)&np.isfinite(im)]
 if pos.size<10:return np.zeros(im.shape,bool)
 lo,hi=np.percentile(pos,[1,99]);x=np.clip((im-lo)/max(hi-lo,1e-8),0,1);x=x+(x-gaussian_filter(x,1));sx,sy=x.shape;search=bm.copy();search[:int(.15*sx),:]=False;search[int(.85*sx):,:]=False;search[:,:int(.15*sy)]=False;search[:,int(.85*sy):]=False;v=x[search&(x>0)];hot=search&(x>=max(otsu(v),float(np.percentile(v,70))));st=generate_binary_structure(2,1);hot=binary_opening(hot,st,1);hot=binary_closing(hot,st,1);comp,n=label(hot,st);mid=sx/2;c=[]
 for k in range(1,n+1):
  m=comp==k;area=int(m.sum());q=np.argwhere(m)
  if area>=3 and area<=.12*search.sum():c.append({'m':m,'area':area,'cx':float(q[:,0].mean()),'cy':float(q[:,1].mean()),'score':float(x[m].mean())*np.sqrt(area)})
 left=[q for q in c if q['cx']<mid];right=[q for q in c if q['cx']>=mid];best=None
 for l in left:
  for r in right:
   score=np.sqrt(l['score']*r['score'])*(0.5+0.5*min(l['area'],r['area'])/max(l['area'],r['area']))*np.exp(-abs(l['cy']-r['cy'])/(.12*sy))*np.exp(-abs(abs(l['cx']-mid)-abs(r['cx']-mid))/(.12*sy))
   if best is None or score>best[0]:best=(score,l['m']|r['m'])
 if best:return best[1]
 crossing=[q for q in c if abs(q['cx']-mid)<=2]
 if crossing:
  m=max(crossing,key=lambda z:z['score'])['m'];l=m.copy();r=m.copy();l[int(mid):,:]=False;r[:int(mid),:]=False
  if l.sum()>=3 and r.sum()>=3:return l|r
 return np.zeros(im.shape,bool)
def shape(mask,spacing=2.):
 yy,xx=np.where(mask)
 if len(xx)<3:return [0.]*6
 area=len(xx)*spacing*spacing;per=(mask&~binary_erosion(mask)).sum()*spacing;circ=4*np.pi*area/max(per*per,1e-12);ev=np.sort(np.linalg.eigvalsh(np.cov(np.column_stack([xx,yy])*spacing,rowvar=False)))[::-1];major=4*np.sqrt(max(ev[0],0));minor=4*np.sqrt(max(ev[-1],0));ecc=np.sqrt(max(0,1-(minor/max(major,1e-12))**2));return [area,per,circ,major,minor,ecc]
def main():
 lab=pd.read_csv(LABELS,dtype={'uid':str}).sort_values('uid').reset_index(drop=True);win=pd.read_csv(WIN,dtype={'uid':str}).set_index('uid'); valid=[]
 for u in lab.uid.astype(str):
  if np.asarray(nib.load(str(BASEMASK/f'{u}_mip_striatum_mask.nii.gz')).dataobj).any():valid.append(u)
 lab=lab[lab.uid.isin(valid)].sort_values('uid').reset_index(drop=True); brain=nib.load(str(ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'));bm=np.asarray(brain.dataobj)>0;y=lab.actual.to_numpy();cv=StratifiedKFold(5,shuffle=True,random_state=SEED); allres=[];allfold=[]
 for sigma in SIGMAS:
  rows=[]
  for u in lab.uid.astype(str):
   a=np.asarray(nib.load(str(REG/f'{u}.nii.gz')).dataobj,dtype=np.float32);s,e=int(win.loc[u,'start']),int(win.loc[u,'end']);slab=a[:,:,s:e+1];mb=bm[:,:,s:e+1];m=segment(slab.max(2),mb.any(2),sigma); left=m&(np.indices(m.shape)[0]>=45);right=m&(np.indices(m.shape)[0]<45);rows.append(shape(left)+shape(right))
  cols=[f'{side}_{feat}' for side in ('left','right') for feat in ('area','perimeter','circularity','major_axis','minor_axis','eccentricity')];X=np.asarray(rows,float);p=np.full(len(y),np.nan)
  for fold,(tr,va) in enumerate(cv.split(X,y),1):
   model=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=SEED+fold));model.fit(X[tr],y[tr]);p[va]=model.predict_proba(X[va])[:,1];q={'sigma':sigma,'fold':fold,'auroc':roc_auc_score(y[va],p[va]),'log_loss':log_loss(y[va],p[va],labels=[0,1]),'accuracy':accuracy_score(y[va],p[va]>=.5),'incorrect':int(np.sum((p[va]>=.5).astype(int)!=y[va]))};allfold.append(q)
  pred=(p>=.5).astype(int);allres.append({'sigma':sigma,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int(np.sum(pred!=y))});print(allres[-1],flush=True)
 OUT.mkdir(parents=True,exist_ok=True);pd.DataFrame(allres).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(allfold).to_csv(OUT/'fold_metrics.csv',index=False);print(pd.DataFrame(allres).to_string(index=False))
if __name__=='__main__':main()
