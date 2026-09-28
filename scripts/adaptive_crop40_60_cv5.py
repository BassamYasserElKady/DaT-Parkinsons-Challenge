#!/usr/bin/env python3
"""Nested five-fold comparison of adaptive 40% and 60% context crops."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

import adaptive_crop30_sbr_cv5 as crop

ROOT=Path(__file__).resolve().parents[1]
LABELS=ROOT/'results/cv5_submission_flip_clip/oof_predictions.csv'
WINDOWS=ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv'
MD=ROOT/'assets/atlases/fsl/masks'
OUT=ROOT/'results/adaptive_crop40_60_cv5'
SEED=20260931
LEVELS=(40,60)

def crop_bounds(mask2d,level,shape):
    p=np.argwhere(mask2d); x0,y0=p.min(0); x1,y1=p.max(0)+1; m=int(round(max(x1-x0,y1-y0)*level/100)); return max(0,x0-m),min(shape[0],x1+m),max(0,y0-m),min(shape[1],y1+m)

def make_masks(template,original,xcoord):
    ca0,pu0,_=crop.base.reposition(template,original); ca=crop.grid.circularize(ca0&(xcoord>=45),.9)|crop.grid.circularize(ca0&(xcoord<45),.9); pu=(crop.grid.shorten(pu0&(xcoord>=45),.6)|crop.grid.shorten(pu0&(xcoord<45),.6))&~ca; return ca,pu

def extract_features(labels,train_idx,level,brain,occ,original,xcoord,windows):
    healthy=labels.iloc[train_idx].loc[labels.iloc[train_idx].actual==0,'uid'].astype(str); total=np.zeros(brain.shape,np.float64)
    for uid in healthy: total+=crop.grid.scaled(uid,brain)
    template=(total/len(healthy)).astype(np.float32); ca,pu=make_masks(template,original,xcoord)
    tx0,tx1,ty0,ty1=crop_bounds((ca|pu).max(2),level,template.shape[:2]); ref,_=crop.adaptive_base.adaptive_localization(template[tx0:tx1,ty0:ty1,:]); refc=(int(ref['start'])+int(ref['end']))/2
    rows=[]
    for uid in labels.uid.astype(str):
        a=crop.grid.scaled(uid,brain); vol=np.asarray(nib.load(str(crop.REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); ist=int(windows.loc[uid,'start']); ien=int(windows.loc[uid,'end']); ish=int(round((ist+ien)/2-refc)); cai=crop.base.translate(ca,np.array([0,0,ish])); pui=crop.base.translate(pu,np.array([0,0,ish])); x0,x1,y0,y1=crop_bounds((cai|pui).max(2),level,vol.shape[:2]); best,_=crop.adaptive_base.adaptive_localization(vol[x0:x1,y0:y1,:]); start,end=int(best['start']),int(best['end']); sh=int(round((start+end)/2-refc)); caf=crop.base.translate(ca,np.array([0,0,sh])); puf=crop.base.translate(pu,np.array([0,0,sh])); gridxy=np.indices(caf.shape); keep=(gridxy[0]>=x0)&(gridxy[0]<x1)&(gridxy[1]>=y0)&(gridxy[1]<y1); caf&=keep; puf&=keep; om=float(a[occ].mean()); cm=float(a[caf].mean()); pm=float(a[puf].mean()); sm=float(a[caf|puf].mean()); rows.append([(cm-om)/om,(pm-om)/om,(sm-om)/om,start,end,sh,int(caf.sum()),int(puf.sum())])
    return pd.DataFrame(rows,columns=['caudate_sbr','putamen_sbr','striatum_sbr','start','end','z_shift','caudate_voxels','putamen_voxels'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lab=pd.read_csv(LABELS,dtype={'uid':str}).sort_values('uid').reset_index(drop=True)
    windows=pd.read_csv(WINDOWS,dtype={'uid':str}).set_index('uid')
    y=lab.actual.to_numpy(dtype=int)
    brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0
    occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0
    names=['left_caudate','right_caudate','left_putamen','right_putamen']
    original={n:np.asarray(nib.load(str(MD/f'{n}.nii.gz')).dataobj)>0 for n in names}
    xcoord=np.indices(brain.shape)[0]
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    oof={level:np.full(len(lab),np.nan) for level in LEVELS}; fold_rows=[]
    for fold,(tr,va) in enumerate(cv.split(lab,y),1):
        for level in LEVELS:
            print(f'fold={fold} crop={level}% starting',flush=True)
            feats=extract_features(lab,tr,level,brain,occ,original,xcoord,windows)
            cols=['caudate_sbr','putamen_sbr','striatum_sbr']
            model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED+fold+level,n_jobs=4)
            model.fit(feats.iloc[tr][cols],y[tr]); p=model.predict_proba(feats.iloc[va][cols])[:,1]; oof[level][va]=p
            fold_rows.append({'fold':fold,'crop_context_pct':level,'n_train':len(tr),'n_validation':len(va),'auroc':roc_auc_score(y[va],p),'log_loss':log_loss(y[va],p,labels=[0,1]),'accuracy':accuracy_score(y[va],p>=.5),'incorrect':int(np.sum((p>=.5).astype(int)!=y[va])),'median_caudate_voxels':float(feats.caudate_voxels.iloc[va].median()),'median_putamen_voxels':float(feats.putamen_voxels.iloc[va].median())})
            print(f"fold={fold} crop={level}% auroc={fold_rows[-1]['auroc']:.6f} logloss={fold_rows[-1]['log_loss']:.6f} accuracy={fold_rows[-1]['accuracy']:.6f}",flush=True)
    rows=[]; frames=[]
    for level,p in oof.items():
        pred=(p>=.5).astype(int); rows.append({'crop_context_pct':level,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int(np.sum(pred!=y))}); frames.append(pd.DataFrame({'uid':lab.uid,'actual':y,'crop_context_pct':level,'oof_probability':p,'oof_prediction':pred}))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame(fold_rows).to_csv(OUT/'fold_metrics.csv',index=False); pd.concat(frames,ignore_index=True).to_csv(OUT/'oof_predictions.csv',index=False)
    print('\nFINAL',flush=True); print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__': main()
