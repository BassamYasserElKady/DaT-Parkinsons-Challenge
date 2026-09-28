#!/usr/bin/env python3
"""Cross-fitted comparison of existing four-model stack versus guided-SVM replacement."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/stack_four_models_inception8_mip/oof_predictions.npz'
NEW=ROOT/'results/nested_cv5_compare_original_svm_vs_mip6_features/oof_predictions.npz'
OUT=ROOT/'results/stack_four_models_inception8_mip_with_guided_svm'
EPS=1e-6
def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p)-np.log1p(-p)
def metric(y,p):
    p=np.clip(p,EPS,1-EPS); return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}
def crossfit(raw,y,fold):
    oof=np.full(len(y),np.nan); records=[]
    for f in sorted(np.unique(fold)):
        tr,te=fold!=f,fold==f; model=LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,random_state=20260931+int(f));model.fit(logit(raw[tr]),y[tr]);p=model.predict_proba(logit(raw[te]))[:,1];oof[te]=p;records.append({'fold':int(f),**metric(y[te],p),'coefficients':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])})
    return oof,records
def main():
    OUT.mkdir(parents=True,exist_ok=True);old=np.load(OLD);new=np.load(NEW)
    if not np.array_equal(old['y'],new['y']): raise RuntimeError('label mismatch')
    expected=np.full(len(old['y']),-1,dtype=int)
    for f,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20260931).split(np.zeros(len(expected)),old['y']),1): expected[te]=f
    if not np.array_equal(old['fold'],expected): raise RuntimeError('fold mismatch')
    y,fold=old['y'].astype(int),old['fold'].astype(int)
    original=np.column_stack((old['resnet'],old['densenet'],old['svm'],old['inception8_mip']))
    replacement=np.column_stack((old['resnet'],old['densenet'],new['plus_mip6_shape_intensity'],old['inception8_mip']))
    p_old,r_old=crossfit(original,y,fold);p_new,r_new=crossfit(replacement,y,fold)
    summary={'evaluation':'cross-fitted five-fold logistic stacking; each meta-model fits four outer folds and predicts the held-out fold','original_stack':metric(y,p_old),'guided_svm_replacement_stack':metric(y,p_new),'standalone_svm':{'original':metric(y,old['svm']),'guided_nested_oof':metric(y,new['plus_mip6_shape_intensity'])},'folds_original':r_old,'folds_guided_svm':r_new}
    np.savez_compressed(OUT/'oof_predictions.npz',y=y,fold=fold,resnet=old['resnet'],densenet=old['densenet'],inception8_mip=old['inception8_mip'],original_svm=old['svm'],guided_svm=new['plus_mip6_shape_intensity'],original_stack=p_old,guided_svm_stack=p_new)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
