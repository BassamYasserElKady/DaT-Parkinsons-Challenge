#!/usr/bin/env python3
"""Cross-validated 10x10 mask-shape search using fold-specific healthy templates."""
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier
import nested_repositioned_mask0_sbr_cv5 as base

R=Path(__file__).resolve().parents[1]; REG=R/'results/ants_registration_mni'; NORM=R/'results/healthy_mode500_tail_gamma_clusters_10/normalized_healthy_volumes'; LAB=R/'results/cv5_submission_flip_clip/oof_predictions.csv'; MD=R/'assets/atlases/fsl/masks'; OUT=R/'results/grid_mask_shape_sbr_cv5'; SEED=20260931; LEVELS=np.arange(10)/10
def scaled(uid,brain):
 p=NORM/f'{uid}.nii.gz'
 if p.exists():return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
 a=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float32);v=a[brain&np.isfinite(a)&(a>0)];c,e=np.histogram(v,bins=500);i=c.argmax();m=(e[i]+e[i+1])/2;z=np.zeros_like(a);q=brain&np.isfinite(a);z[q]=a[q]/m;return z
def circularize(mask,alpha):
 n=int(mask.sum());pts=np.argwhere(mask);center=pts.mean(0);grid=np.indices(mask.shape).astype(float);sphere=-np.sqrt(sum((grid[d]-center[d])**2 for d in range(3)));shape=distance_transform_edt(mask)-distance_transform_edt(~mask);sphere=(sphere-sphere.mean())/(sphere.std()+1e-8);shape=(shape-shape.mean())/(shape.std()+1e-8);score=(1-alpha)*shape+alpha*sphere;idx=np.argpartition(score.ravel(),-n)[-n:];z=np.zeros(mask.size,bool);z[idx]=1;return z.reshape(mask.shape)
def shorten(mask,f):
 if f==0:return mask.copy()
 p=np.argwhere(mask);cut=p[:,1].min()+f*(p[:,1].mean()-p[:,1].min());return mask&(np.indices(mask.shape)[1]>=cut)
def main():
 OUT.mkdir(parents=True,exist_ok=True);lab=pd.read_csv(LAB).sort_values('uid').reset_index(drop=True);y=lab.actual.to_numpy();brain=np.asarray(nib.load(str(MD/'MNI152_T1_2mm_brain_mask.nii.gz')).dataobj)>0;occ=np.asarray(nib.load(str(MD/'occipital_cortex.nii.gz')).dataobj)>0;orig={n:np.asarray(nib.load(str(MD/f'{n}.nii.gz')).dataobj)>0 for n in ['left_caudate','right_caudate','left_putamen','right_putamen']};cv=StratifiedKFold(5,shuffle=True,random_state=SEED);configs=[(ci,pi) for ci in range(10) for pi in range(10)];oof={q:np.full(len(y),np.nan) for q in configs};fold_rows=[]
 for fold,(tr,va) in enumerate(cv.split(lab,y),1):
  hu=lab.iloc[tr].loc[lab.iloc[tr].actual==0,'uid'].astype(str);total=np.zeros(brain.shape,np.float64)
  for uid in hu:total+=scaled(uid,brain)
  base_ca,base_pu,shifts=base.reposition((total/len(hu)).astype(np.float32),orig);X=np.indices(brain.shape)[0];ca=[circularize(base_ca&(X>=45),a)|circularize(base_ca&(X<45),a) for a in LEVELS];pu=[shorten(base_pu&(X>=45),a)|shorten(base_pu&(X<45),a) for a in LEVELS]
  masks=[occ]+ca;pair_masks=[]
  for ci,pi in configs:pair_masks.append(pu[pi]&~ca[ci])
  masks+=pair_masks;mat=sparse.csr_matrix(np.vstack([m.ravel() for m in masks]).astype(np.float32));counts=np.asarray(mat.sum(1)).ravel();values=np.zeros((len(lab),len(masks)),np.float32)
  for i,uid in enumerate(lab.uid.astype(str)):values[i]=np.asarray(mat@scaled(uid,brain).ravel()).ravel()/counts
  om=values[:,0];ca_means=values[:,1:11];pu_means=values[:,11:];
  for qi,(ci,pi) in enumerate(configs):
   cm=ca_means[:,ci];pm=pu_means[:,qi];cn=counts[1+ci];pn=counts[11+qi];sm=(cm*cn+pm*pn)/(cn+pn);feat=np.column_stack(((cm-om)/om,(pm-om)/om,(sm-om)/om));model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED+fold,n_jobs=1);model.fit(feat[tr],y[tr]);oof[(ci,pi)][va]=model.predict_proba(feat[va])[:,1];fold_rows.append({'fold':fold,'caudate_pct':ci*10,'putamen_pct':pi*10,'caudate_voxels':int(cn),'putamen_voxels':int(pn)})
  print(f'fold={fold} complete healthy_reference={len(hu)} configurations=100',flush=True)
 rows=[]
 for (ci,pi),p in oof.items():
  pred=p>=.5;rows.append({'caudate_circularization_pct':ci*10,'putamen_tail_shortening_pct':pi*10,'auroc':roc_auc_score(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'accuracy':accuracy_score(y,pred),'incorrect':int((pred!=y).sum())})
 d=pd.DataFrame(rows).sort_values(['log_loss','auroc'],ascending=[True,False]);d.to_csv(OUT/'grid_metrics.csv',index=False);pd.DataFrame(fold_rows).to_csv(OUT/'fold_mask_sizes.csv',index=False);print(d.head(20).to_string(index=False))
if __name__=='__main__':main()
