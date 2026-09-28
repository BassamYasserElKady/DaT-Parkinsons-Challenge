#!/usr/bin/env python3
"""Exploratory crop-level comparison on an error-informed 80/20 split."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

import grid_mask_shape_sbr_cv5 as grid
import nested_repositioned_mask0_sbr_cv5 as base
import adaptive_slice_best_mask_sbr_cv5 as adaptive_base

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; MD=ROOT/'assets/atlases/fsl/masks'; LABELS=ROOT/'results/cv5_submission_flip_clip/oof_predictions.csv'; PRIOR=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'; RECENT=ROOT/'results/adaptive_crop30_sbr_cv5/oof_predictions.csv'; OUT=ROOT/'results/adaptive_crop_levels_swapped_split'; SEED=20260931
LEVELS=(20,40,60,80)

def crop_bounds(mask2d,level,shape):
 p=np.argwhere(mask2d); x0,y0=p.min(0); x1,y1=p.max(0)+1; m=int(round(max(x1-x0,y1-y0)*level/100)); return max(0,x0-m),min(shape[0],x1+m),max(0,y0-m),min(shape[1],y1+m)

def make_masks(template,orig,xcoord):
 ca0,pu0,_=base.reposition(template,orig); ca=grid.circularize(ca0&(xcoord>=45),.9)|grid.circularize(ca0&(xcoord<45),.9); pu=(grid.shorten(pu0&(xcoord>=45),.6)|grid.shorten(pu0&(xcoord<45),.6))&~ca; return ca,pu

def features(labels,train_idx,level,brain,occ,orig,xcoord,prior):
 healthy=labels.iloc[train_idx].loc[labels.iloc[train_idx].actual==0,'uid'].astype(str); total=np.zeros(brain.shape,np.float64)
 for uid in healthy: total+=grid.scaled(uid,brain)
 template=(total/len(healthy)).astype(np.float32); ca,pu=make_masks(template,orig,xcoord)
 tx0,tx1,ty0,ty1=crop_bounds((ca|pu).max(2),level,template.shape[:2]); ref,_=adaptive_base.adaptive_localization(template[tx0:tx1,ty0:ty1,:]); refc=(int(ref['start'])+int(ref['end']))/2
 rows=[]
 for uid in labels.uid.astype(str):
  a=grid.scaled(uid,brain); vol=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); ist=int(prior.loc[uid,'start']); ien=int(prior.loc[uid,'end']); ish=int(round((ist+ien)/2-refc)); cai=base.translate(ca,np.array([0,0,ish])); pui=base.translate(pu,np.array([0,0,ish])); x0,x1,y0,y1=crop_bounds((cai|pui).max(2),level,vol.shape[:2]); best,_=adaptive_base.adaptive_localization(vol[x0:x1,y0:y1,:]); start,end=int(best['start']),int(best['end']); sh=int(round((start+end)/2-refc)); caf=base.translate(ca,np.array([0,0,sh])); puf=base.translate(pu,np.array([0,0,sh])); gridxy=np.indices(caf.shape); keep=(gridxy[0]>=x0)&(gridxy[0]<x1)&(gridxy[1]>=y0)&(gridxy[1]<y1); caf&=keep; puf&=keep; om=float(a[occ].mean()); cm=float(a[caf].mean()); pm=float(a[puf].mean()); sm=float(a[caf|puf].mean()); rows.append([(cm-om)/om,(pm-om)/om,(sm-om)/om,start,end,sh])
 return pd.DataFrame(rows,columns=['caudate_sbr','putamen_sbr','striatum_sbr','start','end','z_shift'])

def main():
 labels=pd.read_csv(LABELS,dtype={'uid':str}).sort_values('uid').reset_index(drop=True); prior=pd.read_csv(PRIOR,dtype={'uid':str}).set_index('uid'); recent=pd.read_csv(RECENT,dtype={'uid':str}).set_index('uid'); y=labels.actual.to_numpy(); idx=np.arange(len(labels)); tr,te=train_test_split(idx,test_size=.2,stratify=y,random_state=SEED)
 wrong=set(recent.loc[recent.oof_prediction!=recent.actual].index.astype(str)); right=set(recent.loc[recent.oof_prediction==recent.actual].index.astype(str)); test_wrong=[i for i in te if labels.iloc[i].uid in wrong]; train_right=[i for i in tr if labels.iloc[i].uid in right]; n=min(len(test_wrong)//2,len(train_right)); rng=np.random.default_rng(SEED); move_wrong=rng.choice(test_wrong,n,replace=False); move_right=rng.choice(train_right,n,replace=False); tr2=np.array(sorted(set(tr.tolist())|set(move_wrong.tolist())-set(move_right.tolist()))); te2=np.array(sorted(set(te.tolist())-set(move_wrong.tolist())|set(move_right.tolist()))); print(f'initial_train={len(tr)} initial_test={len(te)} wrong_test_candidates={len(test_wrong)} moved_each_way={n} final_train={len(tr2)} final_test={len(te2)}',flush=True)
 brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0; orig={n:np.asarray(nib.load(str(MD/f'{n}.nii.gz')).dataobj)>0 for n in ['left_caudate','right_caudate','left_putamen','right_putamen']}; xcoord=np.indices(brain.shape)[0]; OUT.mkdir(parents=True,exist_ok=True); results=[]
 for level in LEVELS:
  print(f'crop={level}% starting',flush=True); f=features(labels,tr2,level,brain,occ,orig,xcoord,prior); cols=['caudate_sbr','putamen_sbr','striatum_sbr']; model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED+level,n_jobs=4); model.fit(f.iloc[tr2][cols],y[tr2]); p=model.predict_proba(f.iloc[te2][cols])[:,1]; row={'crop_context_pct':level,'n_train':len(tr2),'n_test':len(te2),'auroc':roc_auc_score(y[te2],p),'log_loss':log_loss(y[te2],p,labels=[0,1]),'accuracy':accuracy_score(y[te2],p>=.5),'incorrect':int(np.sum((p>=.5).astype(int)!=y[te2]))}; results.append(row); print(row,flush=True)
 pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame({'uid':labels.iloc[te2].uid,'actual':y[te2]}).to_csv(OUT/'final_test_members.csv',index=False); print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__': main()
