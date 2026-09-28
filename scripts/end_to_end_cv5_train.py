#!/usr/bin/env python3
"""Preprocess, train, and infer all three models after fresh raw-scan registration."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv5_original_flip_vs_dml import (
    DenseNet121_3D, ResNet18CBAM3D, predict, seed_all, train_independent,
)
from cv5_real8_interp16 import standardize_and_resize

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_ensemble_sigma075_8slice_svm"
sys.path.insert(0, str(PACKAGE))
from src.adaptive_localizer import adaptive_localization  # noqa: E402
from src.svm_features import extract_features  # noqa: E402

OUT = ROOT / "results/end_to_end_raw_registration_cv5_seed20260931"
REGISTERED = OUT / "registered"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
BRAIN = ROOT / "assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz"
CNN_INPUT = OUT / "cnn_native8.npy"
SVM_INPUT = OUT / "svm_features22.npy"
SEED = 20260931
WORKERS = 6
EPS = 1e-6


def metric(y, p):
    p = np.clip(np.asarray(p), EPS, 1-EPS)
    return {"auroc": float(roc_auc_score(y,p)), "log_loss": float(log_loss(y,p)),
            "accuracy": float(accuracy_score(y,p>=.5)), "incorrect": int(np.sum((p>=.5)!=y))}


def preprocess_one(payload):
    index, uid = payload
    volume = np.asarray(nib.load(str(REGISTERED/f"{uid}.nii.gz")).dataobj, dtype=np.float64)
    volume = np.where(np.isfinite(volume), np.maximum(volume,0), 0)
    brain = np.asarray(nib.load(str(BRAIN)).dataobj)>0
    svm, svm_audit = extract_features(volume,brain)
    smoothed = ndimage.gaussian_filter(volume,sigma=.75,mode="nearest")
    best,_ = adaptive_localization(smoothed)
    center=(int(best["start"])+int(best["end"]))//2
    start=min(max(center-4,0),smoothed.shape[2]-8);end=start+8
    valid=smoothed>0;positive=smoothed[valid]
    if positive.size==0 or float(positive.std())<1e-8: raise RuntimeError(f"invalid scan {uid}")
    normalized=np.zeros(smoothed.shape,dtype=np.float32)
    normalized[valid]=((smoothed[valid]-float(positive.mean()))/float(positive.std())).astype(np.float32)
    cnn=np.stack([normalized[:,:,z].T for z in range(start,end)],axis=0)
    return index,uid,cnn,svm,{"uid":uid,"cnn_start":start,"cnn_end":end-1,
                              "cnn_center":center,**{f"svm_{k}":v for k,v in svm_audit.items()}}


def build_inputs(rows):
    if CNN_INPUT.exists() and SVM_INPUT.exists() and (OUT/"preprocessing_audit.csv").exists():
        print("reusing_complete_preprocessing",flush=True)
        return np.load(CNN_INPUT),np.load(SVM_INPUT)
    cnn=np.empty((len(rows),8,109,91),dtype=np.float32);svm=np.empty((len(rows),22),dtype=np.float64)
    audits=[None]*len(rows);payloads=[(i,row["uid"]) for i,row in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(preprocess_one,payload) for payload in payloads]
        for count,future in enumerate(as_completed(futures),1):
            index,uid,cnn_row,svm_row,audit=future.result();cnn[index]=cnn_row;svm[index]=svm_row;audits[index]=audit
            if count%100==0 or count==len(rows): print(f"preprocessing={count}/{len(rows)}",flush=True)
    np.save(CNN_INPUT,cnn);np.save(SVM_INPUT,svm);pd.DataFrame(audits).to_csv(OUT/"preprocessing_audit.csv",index=False)
    return cnn,svm


def main():
    all_rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8")))
    transforms=OUT/"transforms"
    rows=[row for row in all_rows if (REGISTERED/f"{row['uid']}.nii.gz").exists()
          and (transforms/f"{row['uid']}.mat").exists()]
    if len(rows)<100:
        raise RuntimeError(f"Too few complete fresh registrations: {len(rows)}")
    pd.DataFrame(rows).to_csv(OUT/"cohort_used.csv",index=False)
    print(f"using_complete_fresh_registration_subset={len(rows)}/{len(all_rows)}",flush=True)
    y=np.asarray([int(row["is_pathologic"]) for row in rows],dtype=int)
    cnn_native,svm_x=build_inputs(rows)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(cnn_native,y))
    fold_id=np.empty(len(y),dtype=int)
    for fold,(_,test) in enumerate(splits,1):fold_id[test]=fold
    progress_path=OUT/"training_progress.json";oof_path=OUT/"oof_progress.npz"
    records=json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed={int(row["fold"]) for row in records}
    if oof_path.exists():
        saved=np.load(oof_path);resnet=saved["resnet"];dense=saved["densenet"];svm_oof=saved["svm"]
    else:
        resnet=np.full(len(y),np.nan);dense=np.full(len(y),np.nan);svm_oof=np.full(len(y),np.nan)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} completed_folds={sorted(completed)}",flush=True)
    for fold,(train,test) in enumerate(splits,1):
        if fold in completed:continue
        seed_all(SEED+fold);x,_,_=standardize_and_resize(cnn_native,train)
        print(f"fold={fold}/5 train={len(train)} test={len(test)}",flush=True)
        r=train_independent(ResNet18CBAM3D(),x,y,train,8,16,1e-3,device,SEED+fold,f"e2e_resnet_f{fold}")
        resnet[test]=predict(r,x,test,device);del r
        if device.type=="cuda":torch.cuda.empty_cache()
        d=train_independent(DenseNet121_3D(),x,y,train,2,14,3e-4,device,SEED+fold,f"e2e_dense_f{fold}")
        dense[test]=predict(d,x,test,device);del d,x
        if device.type=="cuda":torch.cuda.empty_cache()
        s=make_pipeline(StandardScaler(),SVC(C=1,kernel="rbf",gamma="scale",probability=True,random_state=SEED+fold))
        s.fit(svm_x[train],y[train]);svm_oof[test]=s.predict_proba(svm_x[test])[:,1]
        equal=.5*(resnet[test]+dense[test])
        record={"fold":fold,"resnet":metric(y[test],resnet[test]),"densenet":metric(y[test],dense[test]),
                "svm":metric(y[test],svm_oof[test]),"equal_cnn":metric(y[test],equal)}
        records.append(record);records.sort(key=lambda row:row["fold"])
        progress_path.write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
        np.savez(oof_path,resnet=resnet,densenet=dense,svm=svm_oof,fold=fold_id)
        print(json.dumps(record),flush=True)
    if not all(np.all(np.isfinite(p)) for p in (resnet,dense,svm_oof)):raise RuntimeError("Incomplete OOF")
    cnn=.5*(resnet+dense);ensemble=np.full(len(y),np.nan);meta=[]
    for fold in range(1,6):
        train=fold_id!=fold;test=fold_id==fold
        result=minimize_scalar(lambda w:log_loss(y[train],np.clip(w*cnn[train]+(1-w)*svm_oof[train],EPS,1-EPS)),
                               bounds=(0,1),method="bounded",options={"xatol":1e-12})
        w=float(result.x);train_raw=np.clip(w*cnn[train]+(1-w)*svm_oof[train],EPS,1-EPS)
        test_raw=np.clip(w*cnn[test]+(1-w)*svm_oof[test],EPS,1-EPS)
        cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=5000).fit(
            (np.log(train_raw)-np.log1p(-train_raw)).reshape(-1,1),y[train])
        ensemble[test]=cal.predict_proba((np.log(test_raw)-np.log1p(-test_raw)).reshape(-1,1))[:,1]
        meta.append({"fold":fold,"cnn_weight":w,"svm_weight":1-w,
                     "platt_slope":float(cal.coef_[0,0]),"platt_intercept":float(cal.intercept_[0])})
    summary={"n_scans":len(y),"evaluation":"raw NIfTI through fresh registration and complete five-fold inference",
             "registration":"fresh ANTs Rigid using packaged template","resnet":metric(y,resnet),
             "densenet":metric(y,dense),"svm":metric(y,svm_oof),"equal_cnn":metric(y,cnn),
             "three_model_crossfitted":metric(y,ensemble),"folds":records,"meta_parameters":meta}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame({"uid":[row["uid"] for row in rows],"actual":y,"fold":fold_id,"resnet":resnet,
                  "densenet":dense,"svm":svm_oof,"equal_cnn":cnn,"three_model":ensemble}).to_csv(
                      OUT/"oof_predictions.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
