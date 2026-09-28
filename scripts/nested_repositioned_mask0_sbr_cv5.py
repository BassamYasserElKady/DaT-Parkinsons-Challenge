#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np, pandas as pd, nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

R=Path(__file__).resolve().parents[1]; REG=R/'results/ants_registration_mni'; NORM=R/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'; LAB=R/'results/cv5_submission_flip_clip/oof_predictions.csv'; MD=R/'assets/atlases/fsl/masks'; OUT=R/'results/nested_repositioned_mask0_sbr_cv5'
def scaled(uid,brain):
 p=NORM/f'{uid}.nii.gz'
 if p.exists(): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
 a=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); v=a[brain&np.isfinite(a)&(a>0)]; c,e=np.histogram(v,bins=500); i=c.argmax(); m=(e[i]+e[i+1])/2; z=np.zeros_like(a); q=brain&np.isfinite(a); z[q]=a[q]/m; return z
def translate(m,d):
 z=np.zeros_like(m); ss=[]; ds=[]
 for x,L in zip(d,m.shape):
  if x>=0:ss.append(slice(0,L-x));ds.append(slice(x,L))
  else:ss.append(slice(-x,L));ds.append(slice(0,L+x))
 z[tuple(ds)]=m[tuple(ss)]; return z
def reposition(template,orig):
 X,Y=np.indices(template.shape)[:2]; hot=template>=np.percentile(template[template>0],99); guides={}
 for anatomy,side in [('left',X>=45),('right',X<45)]:
  ys=np.argwhere(hot&side)[:,1]; lo,hi=ys.min(),ys.max()
  for structure,bounds in [('caudate',(6,8)),('putamen',(1,5))]:
   g=np.zeros_like(hot)
   for k in range(10):
    if bounds[0]<=k+1<=bounds[1]:g|=hot&side&(Y>=lo+(hi-lo)*k/10)&(Y<=lo+(hi-lo)*(k+1)/10)
   guides[(anatomy,structure)]=g
 placed={}; shifts={}
 for anatomy in ['left','right']:
  for structure in ['caudate','putamen']:
   name=f'{anatomy}_{structure}'; d=np.rint(np.argwhere(guides[(anatomy,structure)]).mean(0)-np.argwhere(orig[name]).mean(0)).astype(int); placed[name]=translate(orig[name],d); shifts[name]=d.tolist()
 return placed['left_caudate']|placed['right_caudate'],placed['left_putamen']|placed['right_putamen'],shifts
def features(rows,brain,occ,ca,pu):
 out=[]
 for _,r in rows.iterrows():
  a=scaled(str(r.uid),brain); o=float(a[occ].mean()); out.append({'uid':str(r.uid),'actual':int(r.actual),'caudate_sbr':(a[ca].mean()-o)/o,'putamen_sbr':(a[pu].mean()-o)/o,'striatum_sbr':(a[ca|pu].mean()-o)/o})
 return pd.DataFrame(out)
def main():
 OUT.mkdir(parents=True,exist_ok=True); lab=pd.read_csv(LAB).sort_values('uid').reset_index(drop=True); ref=nib.load(str(R/'submission_src_ensemble/assets/template.nii')); brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0; orig={n:np.asarray(nib.load(str(MD/f'{n}.nii.gz')).dataobj)>0 for n in ['left_caudate','right_caudate','left_putamen','right_putamen']}; y=lab.actual.to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=20260931); oof={m:np.full(len(y),np.nan) for m in ['svm','xgboost']}; shift_rows=[]
 for fold,(tr,va) in enumerate(cv.split(lab,y),1):
  train=lab.iloc[tr]; hu=train[train.actual==0].uid.astype(str); total=np.zeros(ref.shape,np.float64)
  for uid in hu:total+=scaled(uid,brain)
  ca,pu,shifts=reposition((total/len(hu)).astype(np.float32),orig); tf=features(train,brain,occ,ca,pu); vf=features(lab.iloc[va],brain,occ,ca,pu); cols=['caudate_sbr','putamen_sbr','striatum_sbr']; xx=tf[cols].to_numpy(); zz=vf[cols].to_numpy(); yy=tf.actual.to_numpy()
  for mn in oof:
   model=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',probability=True,random_state=20260931+fold)) if mn=='svm' else XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=20260931+fold,n_jobs=1)
   model.fit(xx,yy); oof[mn][va]=model.predict_proba(zz)[:,1]
  shift_rows.append({'fold':fold,**{k:str(v) for k,v in shifts.items()}}); print(f'fold={fold} healthy_reference={len(hu)} shifts={shifts}',flush=True)
 results=[]; frames=[]
 for mn,p in oof.items():
  pred=(p>=.5).astype(int); results.append({'feature_set':'repositioned_mask0','model':mn,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int((pred!=y).sum())}); frames.append(pd.DataFrame({'uid':lab.uid,'actual':y,'model':mn,'oof_probability':p,'oof_prediction':pred}))
 pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False);pd.concat(frames).to_csv(OUT/'oof_predictions.csv',index=False);pd.DataFrame(shift_rows).to_csv(OUT/'fold_mask_shifts.csv',index=False);(OUT/'summary.json').write_text(json.dumps({'cv':'nested 5-fold','results':results},indent=2));print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
