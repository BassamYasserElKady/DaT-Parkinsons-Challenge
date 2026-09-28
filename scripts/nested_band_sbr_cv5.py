#!/usr/bin/env python3
from pathlib import Path
import os
import json
import numpy as np, pandas as pd, nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

R=Path(__file__).resolve().parents[1]; REG=R/'results/ants_registration_mni'; NORM=R/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'; LABELS=R/'results/cv5_submission_flip_clip/oof_predictions.csv'; MD=R/'assets/atlases/fsl/masks'; OUT=R/os.getenv('NESTED_BAND_OUT','results/nested_band_sbr_cv5'); PUT_BAND_MAX=int(os.getenv('PUT_BAND_MAX','6'))
def scaled(uid,brain):
 p=NORM/f'{uid}.nii.gz'
 if p.exists(): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
 a=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); v=a[brain&np.isfinite(a)&(a>0)]; c,e=np.histogram(v,bins=500); i=c.argmax(); m=(e[i]+e[i+1])/2; z=np.zeros_like(a); q=brain&np.isfinite(a); z[q]=a[q]/m; return z
def bands(template):
 X,Y=np.indices(template.shape)[:2]; hot=template>=np.percentile(template[template>0],99); ca=np.zeros_like(hot); pu=np.zeros_like(hot)
 for side in [X<45,X>=45]:
  ys=np.argwhere(hot&side)[:,1]; lo,hi=ys.min(),ys.max()
  for k in range(10):
   b=hot&side&(Y>=lo+(hi-lo)*k/10)&(Y<=lo+(hi-lo)*(k+1)/10)
   if 6<=k+1<=8: ca|=b
   if 1<=k+1<=PUT_BAND_MAX: pu|=b
 return ca,pu
def make_features(rows,brain,atlas_ca,atlas_pu,occ,bca,bpu):
 out=[]
 for _,r in rows.iterrows():
  a=scaled(str(r.uid),brain); o=float(a[occ].mean())
  out.append({'uid':str(r.uid),'actual':int(r.actual),'atlas_caudate_sbr':(a[atlas_ca].mean()-o)/o,'atlas_putamen_sbr':(a[atlas_pu].mean()-o)/o,'atlas_striatum_sbr':(a[atlas_ca|atlas_pu].mean()-o)/o,'band_caudate_sbr':(a[bca].mean()-o)/o,'band_putamen_sbr':(a[bpu].mean()-o)/o,'band_striatum_sbr':(a[bca|bpu].mean()-o)/o})
 return pd.DataFrame(out)
def main():
 OUT.mkdir(parents=True,exist_ok=True); lab=pd.read_csv(LABELS).sort_values('uid').reset_index(drop=True); ref=nib.load(str(R/'submission_src_ensemble/assets/template.nii')); brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; aca=np.asarray(nib.load(str(MD/'caudate_bilateral.nii.gz')).dataobj)>0; apu=np.asarray(nib.load(str(MD/'putamen_bilateral.nii.gz')).dataobj)>0; occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0; y=lab.actual.to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=20260931); oof={}
 configs={'atlas':['atlas_caudate_sbr','atlas_putamen_sbr','atlas_striatum_sbr'],'band':['band_caudate_sbr','band_putamen_sbr','band_striatum_sbr']}
 for fs,cols in configs.items():
  for mn in ['svm','xgboost']: oof[(fs,mn)]=np.full(len(lab),np.nan)
 for fold,(tr,va) in enumerate(cv.split(lab,y),1):
  train=lab.iloc[tr]; healthy=train[train.actual==0].uid.astype(str).tolist(); total=np.zeros(ref.shape,np.float64)
  for uid in healthy: total += scaled(uid,brain)
  template=(total/len(healthy)).astype(np.float32); bca,bpu=bands(template); trainf=make_features(train,brain,aca,apu,occ,bca,bpu); valf=make_features(lab.iloc[va],brain,aca,apu,occ,bca,bpu)
  for fs,cols in configs.items():
   xx=trainf[cols].to_numpy(); zz=valf[cols].to_numpy(); yy=trainf.actual.to_numpy()
   for mn in ['svm','xgboost']:
    model=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',probability=True,random_state=20260931+fold)) if mn=='svm' else XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=20260931+fold,n_jobs=1)
    model.fit(xx,yy); oof[(fs,mn)][va]=model.predict_proba(zz)[:,1]
  print(f'fold={fold} train={len(tr)} validation={len(va)} healthy_reference={len(healthy)}',flush=True)
 results=[]; outframes=[]
 for (fs,mn),p in oof.items():
  pred=(p>=.5).astype(int); results.append({'feature_set':fs,'model':mn,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int((pred!=y).sum())}); outframes.append(pd.DataFrame({'uid':lab.uid,'actual':y,'feature_set':fs,'model':mn,'oof_probability':p,'oof_prediction':pred}))
 pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); pd.concat(outframes).to_csv(OUT/'oof_predictions.csv',index=False); (OUT/'summary.json').write_text(json.dumps({'cv':'5-fold nested band construction; seed 20260931','healthy_reference_per_fold':'training healthy scans only','results':results},indent=2)); print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
