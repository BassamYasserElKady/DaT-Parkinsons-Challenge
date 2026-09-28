#!/usr/bin/env python3
"""Paper-inspired fixed-slice projection and adaptive-threshold SBR evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter, label, binary_opening, binary_closing, generate_binary_structure
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

import grid_mask_shape_sbr_cv5 as grid

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; NORM=ROOT/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'; LABELS=ROOT/'results/cv5_submission_flip_clip/oof_predictions.csv'; MD=ROOT/'assets/atlases/fsl/masks'; OUT=ROOT/'results/paper_fixed_slices_adaptive_threshold_cv5'; SEED=20260931; SLICE_START=39; SLICE_END=42

def threshold_otsu(values):
 values=np.asarray(values,dtype=np.float64); values=values[np.isfinite(values)]
 if values.size==0 or np.all(values==values[0]): return float(values[0]) if values.size else 0.0
 hist,edges=np.histogram(values,bins=256); centers=(edges[:-1]+edges[1:])/2; prob=hist/hist.sum(); omega=np.cumsum(prob); mu=np.cumsum(prob*centers); total=mu[-1]; between=(total*omega-mu)**2/np.maximum(omega*(1-omega),1e-12); return float(centers[np.nanargmax(between)])

def paper_features(labels,brain,occ):
 rows=[]
 for uid in labels.uid.astype(str):
  a=grid.scaled(uid,brain); slab=a[:,:,SLICE_START:SLICE_END]; proj=np.max(slab,axis=2); pos=proj[np.isfinite(proj)&(proj>0)]
  lo,hi=np.percentile(pos,[1,99]); normalized=np.clip((proj-lo)/max(hi-lo,1e-8),0,1); sharp=normalized+1.0*(normalized-gaussian_filter(normalized,1.0)); search=sharp[np.isfinite(sharp)&(sharp>0)]; cut=float(threshold_otsu(search)) if search.size else 0.0; hot=sharp>=cut; struct=generate_binary_structure(2,1); hot=binary_opening(hot,struct,iterations=1); hot=binary_closing(hot,struct,iterations=1); comp,n=label(hot,struct); mid=hot.shape[0]/2; candidates=[]
  for k in range(1,n+1):
   m=comp==k; area=int(m.sum());
   if area<3: continue
   q=np.argwhere(m); cx=float(q[:,0].mean()); strength=float(sharp[m].mean())*np.sqrt(area); candidates.append((strength,area,cx,m))
  left=[q for q in candidates if q[2]<mid]; right=[q for q in candidates if q[2]>=mid]; left.sort(reverse=True,key=lambda q:q[0]); right.sort(reverse=True,key=lambda q:q[0]); lm=left[0][3] if left else np.zeros(proj.shape,bool); rm=right[0][3] if right else np.zeros(proj.shape,bool); l3=np.repeat(lm[:,:,None],SLICE_END-SLICE_START,axis=2); r3=np.repeat(rm[:,:,None],SLICE_END-SLICE_START,axis=2); om=float(a[occ].mean()); lmval=float(a[:,:,SLICE_START:SLICE_END][l3].mean()) if l3.any() else 0.; rmval=float(a[:,:,SLICE_START:SLICE_END][r3].mean()) if r3.any() else 0.; lefts=(lmval-om)/om; rights=(rmval-om)/om; rows.append([lefts,rights,(lefts+rights)/2,abs(lefts-rights),cut,int(l3.sum()),int(r3.sum()),int(len(candidates))])
 return pd.DataFrame(rows,columns=['left_sbr','right_sbr','mean_sbr','absolute_asymmetry','threshold','left_voxels','right_voxels','component_count'])

def main():
 lab=pd.read_csv(LABELS,dtype={'uid':str}).sort_values('uid').reset_index(drop=True); y=lab.actual.to_numpy(dtype=int); brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0; f=paper_features(lab,brain,occ); cols=['left_sbr','right_sbr','mean_sbr','absolute_asymmetry']; cv=StratifiedKFold(5,shuffle=True,random_state=SEED); p=np.full(len(y),np.nan); fold=[]
 for fold_id,(tr,va) in enumerate(cv.split(lab,y),1):
  model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED+fold_id,n_jobs=4); model.fit(f.iloc[tr][cols],y[tr]); p[va]=model.predict_proba(f.iloc[va][cols])[:,1]; fold.append({'fold':fold_id,'auroc':roc_auc_score(y[va],p[va]),'log_loss':log_loss(y[va],p[va],labels=[0,1]),'accuracy':accuracy_score(y[va],p[va]>=.5),'incorrect':int(np.sum((p[va]>=.5).astype(int)!=y[va]))}); print(fold[-1],flush=True)
 pred=(p>=.5).astype(int); result={'method':'paper_fixed_slices_unsharp_otsu','slices':f'{SLICE_START}:{SLICE_END}','n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int(np.sum(pred!=y))}; OUT.mkdir(parents=True,exist_ok=True); pd.DataFrame([result]).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame(fold).to_csv(OUT/'fold_metrics.csv',index=False); f.insert(0,'uid',lab.uid); f.to_csv(OUT/'paper_features.csv',index=False); pd.DataFrame({'uid':lab.uid,'actual':y,'oof_probability':p,'oof_prediction':pred}).to_csv(OUT/'oof_predictions.csv',index=False); print(pd.DataFrame([result]).to_string(index=False))
if __name__=='__main__': main()
