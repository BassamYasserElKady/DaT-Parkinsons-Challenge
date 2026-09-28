#!/usr/bin/env python3
"""Validate the deployable localizer, cross-validate it, and fit the final SVM/blend."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_ensemble_sigma075_8slice_svm"
sys.path.insert(0, str(PACKAGE))
from src.svm_features import extract_features  # noqa: E402

REG = ROOT / "results/ants_registration_mni"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
BRAIN = ROOT / "assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz"
CNN_OOF = ROOT / "results/cv5_real8_post_registration_gaussian/oof_predictions.csv"
OUT = ROOT / "results/deployable_three_model_svm"
SEED = 20260931
EPS = 1e-6


def metrics(y, p):
    p = np.clip(p, EPS, 1-EPS)
    return {"auroc": float(roc_auc_score(y,p)), "log_loss": float(log_loss(y,p)),
            "accuracy": float(accuracy_score(y,p>=.5)), "incorrect": int(np.sum((p>=.5)!=y))}


def model(fold):
    return make_pipeline(StandardScaler(), SVC(C=1,kernel="rbf",gamma="scale",
                                                probability=True,random_state=SEED+fold))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(LABELS,dtype={"uid":str}).sort_values("uid").reset_index(drop=True)
    y=rows.is_pathologic.to_numpy(dtype=int)
    brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0
    features=[]; audits=[]
    for number,uid in enumerate(rows.uid,1):
        volume=np.asarray(nib.load(str(REG/f"{uid}.nii.gz")).dataobj,dtype=np.float32)
        vector,audit=extract_features(volume,brain)
        features.append(vector); audits.append({"uid":uid,**audit})
        if number%100==0 or number==len(rows): print(f"features={number}/{len(rows)}",flush=True)
    x=np.asarray(features,dtype=np.float64)
    np.save(OUT/"features.npy",x)
    pd.DataFrame(audits).to_csv(OUT/"window_audit.csv",index=False)
    oof=np.full(len(y),np.nan); fold_id=np.empty(len(y),dtype=int); fold_metrics=[]
    splitter=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for fold,(train,test) in enumerate(splitter.split(x,y),1):
        classifier=model(fold); classifier.fit(x[train],y[train])
        oof[test]=classifier.predict_proba(x[test])[:,1]; fold_id[test]=fold
        record={"fold":fold,**metrics(y[test],oof[test])};fold_metrics.append(record)
        print(json.dumps(record),flush=True)
    pd.DataFrame({"uid":rows.uid,"actual":y,"fold":fold_id,"svm":oof}).to_csv(
        OUT/"oof_predictions.csv",index=False)
    final=model(0);final.fit(x,y)
    joblib.dump(final,OUT/"morphometric_intensity_rbf_svm.joblib",compress=3)

    cnn=pd.read_csv(CNN_OOF,dtype={"uid":str}).sort_values("uid").reset_index(drop=True)
    if not np.array_equal(rows.uid.to_numpy(),cnn.uid.to_numpy()): raise RuntimeError("CNN UID mismatch")
    cnn_p=.5*(cnn.resnet.to_numpy()+cnn.densenet.to_numpy())
    blended_oof=np.full(len(y),np.nan);params=[]
    for fold in range(1,6):
        train=fold_id!=fold;test=fold_id==fold
        result=minimize_scalar(lambda w:log_loss(y[train],np.clip(w*cnn_p[train]+(1-w)*oof[train],EPS,1-EPS)),
                               bounds=(0,1),method="bounded",options={"xatol":1e-12})
        weight=float(result.x); train_raw=np.clip(weight*cnn_p[train]+(1-weight)*oof[train],EPS,1-EPS)
        test_raw=np.clip(weight*cnn_p[test]+(1-weight)*oof[test],EPS,1-EPS)
        calibration=LogisticRegression(C=1e6,solver="lbfgs",max_iter=5000)
        calibration.fit((np.log(train_raw)-np.log1p(-train_raw)).reshape(-1,1),y[train])
        blended_oof[test]=calibration.predict_proba(
            (np.log(test_raw)-np.log1p(-test_raw)).reshape(-1,1))[:,1]
        params.append({"fold":fold,"cnn_weight":weight,"svm_weight":1-weight,
                       "platt_slope":float(calibration.coef_[0,0]),
                       "platt_intercept":float(calibration.intercept_[0])})

    global_result=minimize_scalar(lambda w:log_loss(y,np.clip(w*cnn_p+(1-w)*oof,EPS,1-EPS)),
                                  bounds=(0,1),method="bounded",options={"xatol":1e-12})
    cnn_weight=float(global_result.x);raw=np.clip(cnn_weight*cnn_p+(1-cnn_weight)*oof,EPS,1-EPS)
    calibration=LogisticRegression(C=1e6,solver="lbfgs",max_iter=5000)
    calibration.fit((np.log(raw)-np.log1p(-raw)).reshape(-1,1),y)
    metadata={"training_scan_count":len(y),"cnn_weight":cnn_weight,"resnet_weight":cnn_weight/2,
              "densenet_weight":cnn_weight/2,"svm_weight":1-cnn_weight,
              "platt_slope":float(calibration.coef_[0,0]),
              "platt_intercept":float(calibration.intercept_[0]),
              "svm_oof":metrics(y,oof),"three_model_crossfitted_oof":metrics(y,blended_oof),
              "folds":fold_metrics,"meta_fold_parameters":params,
              "localizer":"deployable p95/sigma0.5 five-slice; middle-50 search only when segmentation is empty"}
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame({"uid":rows.uid,"actual":y,"fold":fold_id,"prediction":blended_oof}).to_csv(
        OUT/"three_model_oof_predictions.csv",index=False)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__": main()
