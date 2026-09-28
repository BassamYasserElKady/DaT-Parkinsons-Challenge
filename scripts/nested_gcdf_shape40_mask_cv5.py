#!/usr/bin/env python3
"""Nested cluster-specific GCDF evaluation with the 40% no-overlap mask."""
from pathlib import Path
import json
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import gamma
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier
import nested_repositioned_mask0_sbr_cv5 as maskbase
import refit_gamma_shape_mode_floors as gammabase
from nested_mask0_shape40_overlap_to_caudate_cv5 import reposition_shape40

R=Path(__file__).resolve().parents[1]; REG=R/'results/ants_registration_mni'; NORM=R/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'; LABELS=R/'results/cv5_submission_flip_clip/oof_predictions.csv'; MD=R/'assets/atlases/fsl/masks'; OUT=R/'results/nested_gcdf_shape40_mask_cv5'; SEED=20260931
def scaled(uid,brain):
 p=NORM/f'{uid}.nii.gz'
 if p.exists(): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
 a=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32); v=a[brain&np.isfinite(a)&(a>0)]; c,e=np.histogram(v,bins=500); i=c.argmax(); mode=(e[i]+e[i+1])/2; z=np.zeros_like(a); q=brain&np.isfinite(a); z[q]=a[q]/mode; return z
def healthy_clusters(uids,brain,fold):
 rng=np.random.default_rng(SEED+fold); samples=[]; vals={}; total=np.zeros(brain.shape,np.float64)
 for uid in uids:
  a=scaled(uid,brain); total+=a; v=a[brain&(a>0)&np.isfinite(a)]; vals[uid]=v; samples.append(rng.choice(v,min(10000,len(v)),replace=False))
 sample=np.concatenate(samples); lo,hi=np.percentile(np.log10(sample),[.1,99.9]); edges=np.linspace(lo,hi,129); hist=[]
 for uid in uids:
  c,_=np.histogram(np.clip(np.log10(vals[uid]),lo,hi),bins=edges); hist.append(c/c.sum())
 z=PCA(n_components=.99,svd_solver='full').fit_transform(np.sqrt(np.asarray(hist))); labels=KMeans(10,n_init=50,random_state=SEED+fold).fit_predict(z)
 bg_edges=np.linspace(0,1.1,65); bg=[]
 for uid in uids:
  b=vals[uid][vals[uid]<=1.1]; c,_=np.histogram(b,bins=bg_edges); bg.append(c/max(c.sum(),1))
 bg=np.sqrt(np.asarray(bg)); centroids=np.vstack([bg[labels==k].mean(0) for k in range(10)])
 return labels,vals,centroids,bg_edges,(total/len(uids)).astype(np.float32)
def assign_and_features(rows,brain,centroids,bg_edges,params,ca,pu):
 records=[]
 for _,r in rows.iterrows():
  uid=str(r.uid); a=scaled(uid,brain); v=a[brain&(a>0)&np.isfinite(a)]; b=v[v<=1.1]; c,_=np.histogram(b,bins=bg_edges); h=np.sqrt(c/max(c.sum(),1)); cluster=int(np.argmin(((centroids-h)**2).sum(1))); par=params[cluster]; g=gamma.cdf(np.maximum(a,0),a=par['shape'],scale=par['scale']); records.append({'uid':uid,'actual':int(r.actual),'cluster':cluster,'caudate_gcdf':float(g[ca].mean()),'putamen_gcdf':float(g[pu].mean()),'striatum_gcdf':float(g[ca|pu].mean())})
 return pd.DataFrame(records)
def main():
 OUT.mkdir(parents=True,exist_ok=True); lab=pd.read_csv(LABELS).sort_values('uid').reset_index(drop=True); brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0; orig={n:np.asarray(nib.load(str(MD/f'{n}.nii.gz')).dataobj)>0 for n in ['left_caudate','right_caudate','left_putamen','right_putamen']}; y=lab.actual.to_numpy(); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); oof={m:np.full(len(y),np.nan) for m in ['svm','xgboost']}; fold_params=[]; fold_assign=[]
 for fold,(tr,va) in enumerate(cv.split(lab,y),1):
  train=lab.iloc[tr]; hu=train.loc[train.actual==0,'uid'].astype(str).tolist(); cl,vals,centroids,bgedges,template=healthy_clusters(hu,brain,fold); ca,pu,shifts=reposition_shape40(template,orig); params={}
  for k in range(10):
   x=np.concatenate([vals[u] for u,l in zip(hu,cl) if l==k]); rows,obj=gammabase.fit(x,k); st=[q for q in rows if q['label']=='specific_striatal_uptake'][0]; params[k]=st; fold_params.append({'fold':fold,'cluster':k,'healthy_scans':int((cl==k).sum()),'objective':obj,**{q:st[q] for q in ['mixture_weight','shape','scale','mean','mode']}})
  tf=assign_and_features(train,brain,centroids,bgedges,params,ca,pu); vf=assign_and_features(lab.iloc[va],brain,centroids,bgedges,params,ca,pu); fold_assign.append(vf.assign(fold=fold)); cols=['caudate_gcdf','putamen_gcdf','striatum_gcdf']; xx=tf[cols].to_numpy(); zz=vf[cols].to_numpy(); yy=tf.actual.to_numpy()
  for mn in oof:
   model=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',probability=True,random_state=SEED+fold)) if mn=='svm' else XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED+fold,n_jobs=1)
   model.fit(xx,yy); oof[mn][va]=model.predict_proba(zz)[:,1]
  print(f'fold={fold} complete healthy_reference={len(hu)} cluster_sizes={np.bincount(cl,minlength=10).tolist()}',flush=True)
 results=[]; frames=[]
 for mn,p in oof.items():
  pred=(p>=.5).astype(int); results.append({'feature_set':'cluster_specific_gcdf_shape40_mask','model':mn,'n':len(y),'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int((pred!=y).sum())}); frames.append(pd.DataFrame({'uid':lab.uid,'actual':y,'model':mn,'oof_probability':p,'oof_prediction':pred}))
 pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); pd.concat(frames).to_csv(OUT/'oof_predictions.csv',index=False); pd.DataFrame(fold_params).to_csv(OUT/'fold_gamma_parameters.csv',index=False); pd.concat(fold_assign).to_csv(OUT/'fold_gcdf_features.csv',index=False); (OUT/'summary.json').write_text(json.dumps({'cv':'fully nested five-fold','healthy clustering':'10 histogram clusters per training fold','assignment':'background voxels <=1.1','mask':'40% circularized/shortened, overlap to caudate','results':results},indent=2)); print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
