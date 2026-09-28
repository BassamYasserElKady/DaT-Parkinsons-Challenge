#!/usr/bin/env python3
"""Rebuild guided SVM features from the same rigid cache used at submission."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'results/ants_registration_mni'; LABELS=ROOT/'results/ants_registered_consecutive_resnet18_cbam/split.csv'; BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz'; OUT=ROOT/'results/consistent_guided_svm_auxiliary_stack'; SEED=20260931; EPS=1e-6
sys.path.insert(0,str(ROOT/'submission_ensemble_revised_aux_guided_four_model'))
from src.guided_features import extract_guided_features
from src.svm_features import extract_features

def logit(x):
 x=np.clip(np.asarray(x,float),EPS,1-EPS); return np.log(x)-np.log1p(-x)
def metric(y,p):
 p=np.clip(p,EPS,1-EPS); return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=sorted(csv.DictReader(LABELS.open()),key=lambda r:r['uid']); y=np.asarray([int(r['is_pathologic']) for r in rows]); brain=np.asarray(nib.load(BRAIN).dataobj)>0; x=[]
 for i,row in enumerate(rows,1):
  v=np.asarray(nib.load(REG/f"{row['uid']}.nii.gz").dataobj,dtype=np.float32); base,_=extract_features(v,brain); x.append(np.r_[base,extract_guided_features(v,brain)])
  if i%100==0 or i==len(rows):print(f'features={i}/{len(rows)}',flush=True)
 x=np.asarray(x,float); oof=np.full(len(y),np.nan); folds=np.full(len(y),-1,int); reports=[]
 for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y),1):
  m=make_pipeline(StandardScaler(),SVC(C=10.,kernel='rbf',gamma='scale',probability=True,random_state=SEED+f));m.fit(x[tr],y[tr]);oof[te]=m.predict_proba(x[te])[:,1];folds[te]=f;reports.append({'fold':f,**metric(y[te],oof[te])});print(json.dumps(reports[-1]),flush=True)
 full=make_pipeline(StandardScaler(),SVC(C=10.,kernel='rbf',gamma='scale',probability=True,random_state=SEED));full.fit(x,y);sc=full.named_steps['standardscaler'];sv=full.named_steps['svc'];np.savez_compressed(OUT/'guided_morphometric_rbf_svm.npz',mean=sc.mean_,scale=sc.scale_,support_vectors=sv.support_vectors_,dual_coef=sv.dual_coef_[0],intercept=sv.intercept_[0],gamma=sv._gamma,prob_a=sv.probA_[0],prob_b=sv.probB_[0])
 old=np.load(ROOT/'results/stack_four_models_inception8_mip_with_guided_svm/oof_predictions.npz');aux=np.load(ROOT/'results/cv5_inception8_mip_auxiliary/oof_predictions.npz');assert np.array_equal(old['y'],y) and np.array_equal(aux['y'],y)
 raw=np.c_[old['resnet'],old['densenet'],oof,aux['prediction']]; stack=np.full(len(y),np.nan)
 for f in range(1,6):
  tr=folds!=f;te=~tr;m=LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,random_state=SEED+f).fit(logit(raw[tr]),y[tr]);stack[te]=m.predict_proba(logit(raw[te]))[:,1]
 meta=LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,random_state=SEED).fit(logit(raw),y)
 summary={'consistent_guided_svm_oof':metric(y,oof),'consistent_four_model_stack_oof':metric(y,stack),'folds':reports,'stack_coef':meta.coef_[0].tolist(),'stack_intercept':float(meta.intercept_[0])};np.savez_compressed(OUT/'oof_predictions.npz',uid=np.asarray([r['uid'] for r in rows]),y=y,fold=folds,guided_svm=oof,stack=stack);(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'stack_metadata.json').write_text(json.dumps({'architecture':'resnet18_cbam_3d_densenet121_3d_consistent_guided_82_feature_rbf_svm_auxiliary_scratch_inception_v3_adaptive8_mip_logit_stack','logit_stack':{'models':['resnet','densenet','guided_svm','auxiliary_inception8_mip'],'coef':meta.coef_[0].tolist(),'intercept':float(meta.intercept_[0]),'oof_source':'five-fold leakage-controlled predictions'},'guided_svm':{'feature_count':82,'C':10.0,'registration_cache':'ants_registration_mni'}},indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
