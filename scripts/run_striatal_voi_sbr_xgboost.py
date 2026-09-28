#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'results/ants_registration_mni'
LABELS = ROOT / 'results/ants_registered_consecutive_resnet18_cbam/split.csv'
VOI = ROOT / 'assets/atlases/fsl/masks/striatal_voi_mask_LAS.nii.gz'
OCC = ROOT / 'assets/atlases/fsl/masks/occipital_cortex.nii.gz'
OUT = ROOT / 'results/striatal_voi_sbr_xgboost'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)[['uid', 'is_pathologic']]
    voi_img = nib.load(str(VOI)); voi = voi_img.get_fdata() > 0
    occ_img = nib.load(str(OCC)); occ = occ_img.get_fdata() > 0
    rows = []
    for _, r in labels.iterrows():
        uid = str(r.uid); p = REG / f'{uid}.nii.gz'
        if not p.exists(): continue
        img = nib.load(str(p))
        if img.shape != voi_img.shape or not np.allclose(img.affine, voi_img.affine):
            continue
        a = img.get_fdata().astype(np.float32)
        s, o = a[voi], a[occ]
        s, o = s[np.isfinite(s)], o[np.isfinite(o)]
        sm, om = float(s.mean()), float(o.mean())
        sbr = float((sm - om) / om) if om > 0 else np.nan
        rows.append({'uid': uid, 'actual': int(r.is_pathologic), 'striatal_mean': sm, 'occipital_mean': om, 'striatal_sbr': sbr})
    d = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    d.to_csv(OUT/'features.csv', index=False)
    X = d[['striatal_sbr']].to_numpy(); y = d.actual.to_numpy(); p = np.full(len(y), np.nan)
    cv = StratifiedKFold(5, shuffle=True, random_state=20260931)
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.9, colsample_bytree=1.0, reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss', random_state=20260931+fold, n_jobs=1)
        model.fit(X[tr], y[tr]); p[va] = model.predict_proba(X[va])[:, 1]
    pred = (p >= .5).astype(int)
    metrics = {'model':'xgboost', 'feature':'bilateral striatal VOI SBR', 'n':int(len(y)), 'auroc':float(roc_auc_score(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])), 'accuracy':float(accuracy_score(y,pred)), 'incorrect':int((pred != y).sum())}
    out = d[['uid','actual','striatal_mean','occipital_mean','striatal_sbr']].copy(); out['oof_probability']=p; out['fold_prediction']=(p>=.5).astype(int); out.to_csv(OUT/'oof_predictions.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2)); print(f'output={OUT}')

if __name__ == '__main__': main()
