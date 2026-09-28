#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'results/ants_registration_mni'; NORM=ROOT/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'
LABELS=ROOT/'results/cv5_submission_flip_clip/oof_predictions.csv'; MASKDIR=ROOT/'assets/atlases/fsl/masks'; OUT=ROOT/'results/band_vs_atlas_sbr_cv5'

def scale_raw(uid, brain):
    p=NORM/f'{uid}.nii.gz'
    if p.exists(): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
    img=nib.load(str(REG/f'{uid}.nii.gz')); a=np.asarray(img.dataobj,dtype=np.float32); pos=a[brain&np.isfinite(a)&(a>0)]; c,e=np.histogram(pos,bins=500); i=c.argmax(); mode=(e[i]+e[i+1])/2; out=np.zeros_like(a); inside=brain&np.isfinite(a); out[inside]=a[inside]/mode; return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); labels=pd.read_csv(LABELS).sort_values('uid')
    brain=np.asarray(nib.load(str(MASKDIR/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0
    ca=np.asarray(nib.load(str(MASKDIR/'caudate_bilateral.nii.gz')).dataobj)>0; pu=np.asarray(nib.load(str(MASKDIR/'putamen_bilateral.nii.gz')).dataobj)>0
    occ=np.asarray(nib.load(str(MASKDIR/'occipital_cortex.nii.gz')).dataobj)>0
    pop=np.asarray(nib.load(str(ROOT/'results/healthy_empirical_template_mode500/healthy_mean_template.nii.gz')).dataobj); X,Y=np.indices(pop.shape)[:2]; hot=pop>=np.percentile(pop[pop>0],99); bca=np.zeros_like(hot); bpu=np.zeros_like(hot)
    for side in [X<45,X>=45]:
        ys=np.argwhere(hot&side)[:,1]; lo,hi=ys.min(),ys.max()
        for k in range(10):
            y0=lo+(hi-lo)*k/10; y1=lo+(hi-lo)*(k+1)/10; band=hot&side&(Y>=y0)&(Y<=y1)
            if 6<=k+1<=8: bca|=band
            if 1<=k+1<=6: bpu|=band
    rows=[]
    for _,r in labels.iterrows():
        uid=str(r.uid); a=scale_raw(uid,brain); o=float(a[occ].mean());
        if o<=0: continue
        vals={'uid':uid,'actual':int(r.actual),'atlas_caudate_sbr':float((a[ca].mean()-o)/o),'atlas_putamen_sbr':float((a[pu].mean()-o)/o),'atlas_striatum_sbr':float((a[ca|pu].mean()-o)/o),'band_caudate_sbr':float((a[bca].mean()-o)/o),'band_putamen_sbr':float((a[bpu].mean()-o)/o),'band_striatum_sbr':float((a[bca|bpu].mean()-o)/o)}; rows.append(vals)
    d=pd.DataFrame(rows); d.to_csv(OUT/'features.csv',index=False); y=d.actual.to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=20260931); results=[]; oofs=[]
    sets={'atlas':['atlas_caudate_sbr','atlas_putamen_sbr','atlas_striatum_sbr'],'band':['band_caudate_sbr','band_putamen_sbr','band_striatum_sbr']}
    for fs,cols in sets.items():
        Xf=d[cols].to_numpy()
        for model_name in ['svm','xgboost']:
            p=np.full(len(y),np.nan)
            for fold,(tr,va) in enumerate(cv.split(Xf,y),1):
                if model_name=='svm': model=make_pipeline(StandardScaler(),SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=20260931+fold))
                else: model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1.0,reg_lambda=1.0,objective='binary:logistic',eval_metric='logloss',random_state=20260931+fold,n_jobs=1)
                model.fit(Xf[tr],y[tr]); p[va]=model.predict_proba(Xf[va])[:,1]
            pred=(p>=.5).astype(int); row={'feature_set':fs,'model':model_name,'n':len(y),'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,pred)),'incorrect':int((pred!=y).sum())}; results.append(row); oofs.append(pd.DataFrame({'uid':d.uid,'actual':y,'feature_set':fs,'model':model_name,'oof_probability':p,'oof_prediction':pred}))
    pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); pd.concat(oofs,ignore_index=True).to_csv(OUT/'oof_predictions.csv',index=False); (OUT/'summary.json').write_text(json.dumps({'n_scans':len(d),'cv':'5-fold stratified, seed 20260931','results':results},indent=2)+'\n'); print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__': main()
