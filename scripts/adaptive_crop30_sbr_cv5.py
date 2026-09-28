#!/usr/bin/env python3
"""Five-fold evaluation with a 30% contextual crop before adaptive localization."""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

import grid_mask_shape_sbr_cv5 as grid
import nested_repositioned_mask0_sbr_cv5 as base
import adaptive_slice_best_mask_sbr_cv5 as adaptive_base

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
LABELS = ROOT / 'results/cv5_submission_flip_clip/oof_predictions.csv'
MD = ROOT / 'assets/atlases/fsl/masks'
OUT = ROOT / 'results/adaptive_crop30_sbr_cv5'
SEED = 20260931


def bounds(mask2d, level, shape):
    p = np.argwhere(mask2d)
    x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0) + 1
    margin = int(round(max(x1-x0, y1-y0) * level / 100.0))
    return max(0, x0-margin), min(shape[0], x1+margin), max(0, y0-margin), min(shape[1], y1+margin)


def translated_masks(template, original, xcoord):
    ca0, pu0, shifts = base.reposition(template, original)
    ca = grid.circularize(ca0 & (xcoord >= 45), .9) | grid.circularize(ca0 & (xcoord < 45), .9)
    pu = (grid.shorten(pu0 & (xcoord >= 45), .6) | grid.shorten(pu0 & (xcoord < 45), .6)) & ~ca
    return ca, pu, shifts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(LABELS, dtype={'uid': str}).sort_values('uid').reset_index(drop=True)
    y = lab.actual.to_numpy(dtype=int)
    brain = np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj) > 0
    occ = np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj) > 0
    names = ['left_caudate','right_caudate','left_putamen','right_putamen']
    original = {n: np.asarray(nib.load(str(MD/f'{n}.nii.gz')).dataobj)>0 for n in names}
    xcoord = np.indices(brain.shape)[0]
    prior = pd.read_csv(ROOT/'results/adaptive_slice_best_mask_sbr_cv5/adaptive_windows.csv', dtype={'uid':str}).set_index('uid')
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(lab), np.nan)
    diag=[]
    for fold,(tr,va) in enumerate(cv.split(lab,y),1):
        healthy = lab.iloc[tr].loc[lab.iloc[tr].actual==0,'uid'].astype(str)
        total=np.zeros(brain.shape,np.float64)
        for uid in healthy: total += grid.scaled(uid,brain)
        template=(total/len(healthy)).astype(np.float32)
        ca,pu,_=translated_masks(template,original,xcoord)
        base_bounds=bounds((ca|pu).max(axis=2),30,template.shape[:2])
        tx0,tx1,ty0,ty1=base_bounds
        template_crop=template[tx0:tx1,ty0:ty1,:]
        ref,_=adaptive_base.adaptive_localization(template_crop)
        ref_center=(int(ref['start'])+int(ref['end']))/2
        rows=[]
        for uid in lab.uid.astype(str):
            volume=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32)
            # Initial localizer determines the scan-specific mask shift used to define its crop.
            initial_start=int(prior.loc[uid,'start']); initial_end=int(prior.loc[uid,'end'])
            initial_shift=int(round((initial_start+initial_end)/2-ref_center))
            ca0=base.translate(ca,np.array([0,0,initial_shift])); pu0=base.translate(pu,np.array([0,0,initial_shift]))
            x0,x1,y0,y1=bounds((ca0|pu0).max(axis=2),30,volume.shape[:2])
            cropped=volume[x0:x1,y0:y1,:]
            best,_=adaptive_base.adaptive_localization(cropped)
            start,end=int(best['start']),int(best['end'])
            shift=int(round((start+end)/2-ref_center))
            ca1=base.translate(ca,np.array([0,0,shift])); pu1=base.translate(pu,np.array([0,0,shift]))
            # The crop is centered on the ROI, so all retained ROI voxels remain inside it.
            ca1 &= np.zeros_like(ca1,dtype=bool) | ((np.indices(ca1.shape)[0]>=x0)&(np.indices(ca1.shape)[0]<x1)&(np.indices(ca1.shape)[1]>=y0)&(np.indices(ca1.shape)[1]<y1))
            pu1 &= np.zeros_like(pu1,dtype=bool) | ((np.indices(pu1.shape)[0]>=x0)&(np.indices(pu1.shape)[0]<x1)&(np.indices(pu1.shape)[1]>=y0)&(np.indices(pu1.shape)[1]<y1))
            om=float(grid.scaled(uid,brain)[occ].mean()); a=grid.scaled(uid,brain)
            cm=float(a[ca1].mean()); pm=float(a[pu1].mean()); sm=float(a[ca1|pu1].mean())
            rows.append([(cm-om)/om,(pm-om)/om,(sm-om)/om,start,end,shift,x0,x1,y0,y1,int(ca1.sum()),int(pu1.sum())])
        f=pd.DataFrame(rows,columns=['caudate_sbr','putamen_sbr','striatum_sbr','start','end','z_shift','x0','x1','y0','y1','caudate_voxels','putamen_voxels'])
        cols=['caudate_sbr','putamen_sbr','striatum_sbr']; model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED+fold,n_jobs=4)
        model.fit(f.iloc[tr][cols],y[tr]); p=model.predict_proba(f.iloc[va][cols])[:,1]; oof[va]=p
        diag.append({'fold':fold,'reference_start':int(ref['start']),'reference_end':int(ref['end']),'crop_x0':tx0,'crop_x1':tx1,'crop_y0':ty0,'crop_y1':ty1,'median_shift':float(f.z_shift.median()),'min_shift':int(f.z_shift.min()),'max_shift':int(f.z_shift.max()),'median_ca_voxels':float(f.caudate_voxels.median()),'median_pu_voxels':float(f.putamen_voxels.median())})
        print(f'fold={fold} auroc={roc_auc_score(y[va],p):.6f} logloss={log_loss(y[va],p,labels=[0,1]):.6f} accuracy={accuracy_score(y[va],p>=.5):.6f}',flush=True)
    pred=(oof>=.5).astype(int); result={'method':'adaptive_crop30','n':len(y),'auroc':roc_auc_score(y,oof),'log_loss':log_loss(y,oof,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int((pred!=y).sum())}
    pd.DataFrame([result]).to_csv(OUT/'metrics.csv',index=False); pd.DataFrame({'uid':lab.uid,'actual':y,'oof_probability':oof,'oof_prediction':pred}).to_csv(OUT/'oof_predictions.csv',index=False); pd.DataFrame(diag).to_csv(OUT/'fold_diagnostics.csv',index=False)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__=='__main__': main()
