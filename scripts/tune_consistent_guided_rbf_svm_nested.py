#!/usr/bin/env python3
"""Nested search of RBF-SVM hyperparameters on fixed consistent 82 features."""
import csv,json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
ROOT=Path(__file__).resolve().parents[1];REG=ROOT/'results/ants_registration_mni';LABELS=ROOT/'results/ants_registered_consecutive_resnet18_cbam/split.csv';BRAIN=ROOT/'assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz';OUT=ROOT/'results/tune_consistent_guided_rbf_svm_nested';SEED=20260931
sys.path.insert(0,str(ROOT/'submission_ensemble_revised_aux_guided_four_model'))
from src.guided_features import extract_guided_features
from src.svm_features import extract_features
def met(y,p):return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}
def main():
 OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'features.npy';rows=sorted(csv.DictReader(LABELS.open()),key=lambda r:r['uid']);y=np.asarray([int(r['is_pathologic']) for r in rows])
 if cache.exists(): x=np.load(cache)
 else:
  brain=np.asarray(nib.load(BRAIN).dataobj)>0;x=[]
  for i,r in enumerate(rows,1):
   v=np.asarray(nib.load(REG/f"{r['uid']}.nii.gz").dataobj,dtype=np.float32);base,_=extract_features(v,brain);x.append(np.r_[base,extract_guided_features(v,brain)])
   if i%100==0 or i==len(rows):print(f'features={i}/{len(rows)}',flush=True)
  x=np.asarray(x,float);np.save(cache,x)
 n=x.shape[1];params={'svc__C':[.03,.1,.3,1.,3.,10.,30.,100.],'svc__gamma':[.125/n,.25/n,.5/n,1/n,2/n,4/n,8/n],'svc__class_weight':[None,'balanced']};outer=StratifiedKFold(5,shuffle=True,random_state=SEED);p=np.full(len(y),np.nan);report=[]
 for fold,(tr,te) in enumerate(outer.split(x,y),1):
  pipe=Pipeline([('scale',StandardScaler()),('svc',SVC(kernel='rbf',probability=True,random_state=SEED+fold))]);inner=StratifiedKFold(3,shuffle=True,random_state=SEED+fold)
  search=GridSearchCV(pipe,params,scoring='neg_log_loss',cv=inner,n_jobs=-1,refit=True,return_train_score=False);search.fit(x[tr],y[tr]);p[te]=search.predict_proba(x[te])[:,1];row={'fold':fold,'best':search.best_params_,**met(y[te],p[te])};report.append(row);(OUT/'progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
 summary={'evaluation':'nested five-fold; RBF SVM feature vector fixed at 82; inner selection minimizes log loss','grid':{'C':params['svc__C'],'gamma_multiples_of_1_over_features':[.125,.25,.5,1,2,4,8],'class_weight':params['svc__class_weight']},'pooled':met(y,p),'folds':report};np.savez_compressed(OUT/'oof_predictions.npz',y=y,prediction=p);(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
