from pathlib import Path
import json
import numpy as np, pandas as pd, nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

R=Path(__file__).resolve().parents[1]; REG=R/'results/ants_registration_mni'; NORM=R/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'; LABELS=R/'results/cv5_submission_flip_clip/oof_predictions.csv'; MD=R/'assets/atlases/fsl/masks'; OUT=R/'results/band_vs_atlas_sbr_cv5'
def scaled(uid,brain):
 p=NORM/f'{uid}.nii.gz'
 if p.exists(): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
 a=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); v=a[brain&np.isfinite(a)&(a>0)]; c,e=np.histogram(v,bins=500); i=c.argmax(); m=(e[i]+e[i+1])/2; z=np.zeros_like(a); q=brain&np.isfinite(a); z[q]=a[q]/m; return z
def main():
 OUT.mkdir(parents=True,exist_ok=True); lab=pd.read_csv(LABELS).sort_values('uid'); brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; ca=np.asarray(nib.load(str(MD/'caudate_bilateral.nii.gz')).dataobj)>0; pu=np.asarray(nib.load(str(MD/'putamen_bilateral.nii.gz')).dataobj)>0; occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0
 pop=np.asarray(nib.load(str(R/'results/healthy_empirical_template_mode500/healthy_mean_template.nii.gz')).dataobj); X,Y=np.indices(pop.shape)[:2]; hot=pop>=np.percentile(pop[pop>0],99); bca=np.zeros_like(hot); bpu=np.zeros_like(hot)
 for side in [X<45,X>=45]:
  ys=np.argwhere(hot&side)[:,1]; lo,hi=ys.min(),ys.max()
  for k in range(10):
   band=hot&side&(Y>=lo+(hi-lo)*k/10)&(Y<=lo+(hi-lo)*(k+1)/10)
   if 6<=k+1<=8:bca|=band
   if 1<=k+1<=6:bpu|=band
 rows=[]
 for _,r in lab.iterrows():
  a=scaled(str(r.uid),brain); o=float(a[occ].mean())
  if o>0: rows.append({'uid':str(r.uid),'actual':int(r.actual),'atlas_caudate_sbr':(a[ca].mean()-o)/o,'atlas_putamen_sbr':(a[pu].mean()-o)/o,'atlas_striatum_sbr':(a[ca|pu].mean()-o)/o,'band_caudate_sbr':(a[bca].mean()-o)/o,'band_putamen_sbr':(a[bpu].mean()-o)/o,'band_striatum_sbr':(a[bca|bpu].mean()-o)/o})
 d=pd.DataFrame(rows); d.to_csv(OUT/'features.csv',index=False); y=d.actual.to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=20260931); results=[]; oo=[]
 for fs,cols in {'atlas':['atlas_caudate_sbr','atlas_putamen_sbr','atlas_striatum_sbr'],'band':['band_caudate_sbr','band_putamen_sbr','band_striatum_sbr']}.items():
  xx=d[cols].to_numpy()
  for mn in ['svm','xgboost']:
   p=np.full(len(y),np.nan)
   for fold,(tr,va) in enumerate(cv.split(xx,y),1):
    model=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',probability=True,random_state=20260931+fold)) if mn=='svm' else XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=20260931+fold,n_jobs=1)
    model.fit(xx[tr],y[tr]); p[va]=model.predict_proba(xx[va])[:,1]
   pred=(p>=.5).astype(int); results.append({'feature_set':fs,'model':mn,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int((pred!=y).sum())}); oo.append(pd.DataFrame({'uid':d.uid,'actual':y,'feature_set':fs,'model':mn,'oof_probability':p,'oof_prediction':pred}))
 pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); pd.concat(oo).to_csv(OUT/'oof_predictions.csv',index=False); (OUT/'summary.json').write_text(json.dumps({'n_scans':len(d),'results':results},indent=2)); print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
