#!/usr/bin/env python3
"""Nested 5-fold evaluation of original and MIP6 augmented RBF-SVM features."""
from __future__ import annotations

import csv, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from holdout80_fixed299_four_model import build, LABELS, SEED
SHAPES=ROOT/'results/adaptive8_mip6_guided_3d_quarter_shapes/labels'
MAPS=ROOT/'results/adaptive8_mip6_guided_sliding_intensity_pyramid/maps'
OUT=ROOT/'results/nested_cv5_compare_original_svm_vs_mip6_features'
QS=(.05,.25,.5,.75,.95); C_GRID=(.1,1.,10.)

def metric(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}

def extra(uid):
    label=np.asarray(nib.load(str(SHAPES/f'{uid}_adaptive8_3d_quarters.nii.gz')).dataobj,dtype=np.uint8)
    left=np.asarray([np.sum(label==i) for i in range(1,5)],float); right=np.asarray([np.sum(label==i) for i in range(5,9)],float)
    out=[left.sum(),right.sum(),*left,*right,*(left/max(left.sum(),1.)),*(right/max(right.sum(),1.))]
    a=np.load(MAPS/f'{uid}_adaptive8_sliding_pyramid.npz'); raw=a['intensity_1'].astype(float); pos=raw[raw>0]; scale=max(float(np.median(pos)) if len(pos) else 1.,1e-6)
    for side in (0,1):
        use=a['side']==side
        for key in ('intensity_1','intensity_2','intensity_3'):
            v=a[key][use].astype(float)/scale
            out += [v.mean(),v.std(),*np.quantile(v,QS)] if len(v) else [0.]*7
    return np.asarray(out,float)

def model(c,seed): return make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',probability=True,random_state=seed))

def select_c(x,y,train,seed):
    inner=StratifiedKFold(3,shuffle=True,random_state=seed); losses=[]
    for c in C_GRID:
        p=np.full(len(train),np.nan)
        for a,b in inner.split(train,y[train]):
            fit,val=train[a],train[b]; m=model(c,seed+int(c*100));m.fit(x[fit],y[fit]);p[b]=m.predict_proba(x[val])[:,1]
        losses.append(float(log_loss(y[train],p,labels=[0,1])))
    return float(C_GRID[int(np.argmin(losses))]), losses

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=list(csv.DictReader(LABELS.open()));uids=[r['uid'] for r in rows];y=np.asarray([int(r['is_pathologic']) for r in rows])
    _,_,base=build(rows); added=np.asarray([extra(uid) for uid in uids]); arrays={'original22':base,'plus_mip6_shape_intensity':np.column_stack((base,added))}
    oof={name:np.full(len(y),np.nan) for name in arrays}; outer=StratifiedKFold(5,shuffle=True,random_state=SEED); folds=[]
    for fold,(train,test) in enumerate(outer.split(base,y),1):
        row={'fold':fold}
        for name,x in arrays.items():
            c,inner_losses=select_c(x,y,train,SEED+fold)
            m=model(c,SEED+fold);m.fit(x[train],y[train]);oof[name][test]=m.predict_proba(x[test])[:,1]
            row[name]={'selected_C':c,'inner_log_loss_by_C':dict(zip(map(str,C_GRID),inner_losses)),'outer':metric(y[test],oof[name][test])}
        folds.append(row);print(json.dumps(row),flush=True)
    summary={'evaluation':'nested 5-fold; scaler, SVM, and C selection are fit using outer-training data only','n_scans':len(y),'features':{'original22':22,'plus_mip6_shape_intensity':int(arrays['plus_mip6_shape_intensity'].shape[1])},'C_grid':C_GRID,'pooled':{n:metric(y,p) for n,p in oof.items()},'folds':folds}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(OUT/'oof_predictions.npz',uid=np.asarray(uids),y=y,original22=oof['original22'],plus_mip6_shape_intensity=oof['plus_mip6_shape_intensity']);print(json.dumps(summary['pooled'],indent=2),flush=True)
if __name__=='__main__':main()
