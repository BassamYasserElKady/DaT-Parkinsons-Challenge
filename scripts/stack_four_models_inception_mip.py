#!/usr/bin/env python3
"""Leakage-controlled four-model logit stack with Inception-v3 MIP."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; LABELS=ROOT/"results/ants_registered_consecutive_resnet18_cbam/split.csv"; INC=ROOT/"results/cv5_inception_v3_adaptive5_mip/oof_predictions.npz"; OUT=ROOT/"results/stack_four_models_inception_mip"; EPS=1e-6
def logit(p):
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS); return np.log(p)-np.log1p(-p)
def metric(y,p):
    p=np.clip(p,EPS,1-EPS); return {"auroc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p)),"accuracy":float(accuracy_score(y,p>=.5)),"incorrect":int(np.sum((p>=.5)!=y))}
def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); uids=[r["uid"] for r in rows]; y=np.asarray([int(r["is_pathologic"]) for r in rows],dtype=int); n=len(y); base=np.full((n,3),np.nan); folds=np.full(n,-1)
    for fold in range(1,6):
        a=np.load(ROOT/f"results/fold{fold}_five_slice_no_gaussian_three_model_nested/predictions.npz"); by={str(u):i for i,u in enumerate(a["uid"])}
        for j,uid in enumerate(uids):
            if uid in by:
                i=by[uid]; base[j]=[float(a["resnet"][i]),float(a["densenet"][i]),float(a["svm"][i])]; folds[j]=fold
    ia=np.load(INC); inception=np.asarray(ia["prediction"],dtype=float)
    if len(inception)!=n or not np.array_equal(np.asarray(ia["y"],dtype=int),y) or np.any(folds<0) or np.any(~np.isfinite(base)):
        raise RuntimeError("prediction alignment failed")
    raw=np.column_stack((base,inception)); oof=np.full(n,np.nan); rec=[]
    for fold in range(1,6):
        tr=folds!=fold; te=folds==fold; meta=LogisticRegression(C=1.0,solver="lbfgs",max_iter=5000,random_state=20260931+fold); meta.fit(logit(raw[tr]),y[tr]); p=meta.predict_proba(logit(raw[te]))[:,1]; oof[te]=p; rec.append({"fold":fold,**metric(y[te],p),"coefficients":meta.coef_[0].tolist(),"intercept":float(meta.intercept_[0])})
    summary={"models":["ResNet-18","DenseNet-121","RBF-SVM","scratch Inception-v3 adaptive-five-slice MIP"],"calibration":"leakage-controlled five-fold logistic regression on model logits","folds":rec,"pooled":metric(y,oof),"standalone":{"resnet":metric(y,base[:,0]),"densenet":metric(y,base[:,1]),"svm":metric(y,base[:,2]),"inception_mip":metric(y,inception)}}
    np.savez(OUT/"oof_predictions.npz",y=y,resnet=base[:,0],densenet=base[:,1],svm=base[:,2],inception_mip=inception,stack=oof,fold=folds); (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
