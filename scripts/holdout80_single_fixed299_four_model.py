#!/usr/bin/env python3
"""Single clean stratified 80/20 holdout for the repaired four-model pipeline."""
from __future__ import annotations
import csv, json, traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from holdout80_fixed299_four_model import (
    ROOT, OUT as BASE_OUT, SEED, EPS, LABELS, build, cnn_data, train3d,
    predict3d, ResNet18CBAM3D, DenseNet121_3D, Inc, inc_x, metrics, logit,
)

OUT=ROOT/'results/holdout80_single_fixed299_four_model'

def save_progress(stage, **values):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'progress.json').write_text(json.dumps({'stage':stage,**values},indent=2)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(LABELS.open())); y=np.asarray([int(r['is_pathologic']) for r in rows])
    # Inputs originate from the verified cache of all 1,362 raw ANTs-rigid registrations.
    five,mip,svm=build(rows)
    dev,test=next(StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=SEED).split(five,y))
    dev,test=np.sort(dev),np.sort(test); np.savez(OUT/'split.npz',development=dev,test=test,y=y)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); save_progress('inputs_ready',development=len(dev),test=len(test),device=str(device))
    x=cnn_data(five,dev); mx=inc_x(mip); final=np.empty((len(test),4),float)
    r=train3d(ResNet18CBAM3D(),x,y,dev,8,16,1e-3,SEED+1,device,'resnet');final[:,0]=predict3d(r,x,test,device);del r;torch.cuda.empty_cache();save_progress('resnet_complete',metrics=metrics(y[test],final[:,0]))
    d=train3d(DenseNet121_3D(),x,y,dev,2,14,3e-4,SEED+2,device,'densenet');final[:,1]=predict3d(d,x,test,device);del d;torch.cuda.empty_cache();save_progress('densenet_complete',metrics=metrics(y[test],final[:,1]))
    s=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=SEED+3));s.fit(svm[dev],y[dev]);final[:,2]=s.predict_proba(svm[test])[:,1];save_progress('svm_complete',metrics=metrics(y[test],final[:,2]))
    i=train3d(Inc(),mx,y,dev,16,16,1e-3,SEED+4,device,'inception',is_2d=True);final[:,3]=predict3d(i,mx,test,device,is_2d=True);del i;torch.cuda.empty_cache();save_progress('inception_complete',metrics=metrics(y[test],final[:,3]))
    # Use the repaired package's fixed full-data logit-stack parameters, exactly as deployed.
    md=json.loads((ROOT/'submission_ensemble_four_model_logit_stack/assets/model_metadata.json').read_text()); st=md['logit_stack']; stack=1/(1+np.exp(-np.clip(logit(final)@np.asarray(st['coef'])+float(st['intercept']),-40,40)))
    summary={'evaluation':'one stratified 80/20 holdout; one training pass per model','registration_cache':'results/end_to_end_raw_registration_cv5_seed20260931/registered (1,362 raw ANTs rigid registrations)','development_scans':int(len(dev)),'test_scans':int(len(test)),'resnet':metrics(y[test],final[:,0]),'densenet':metrics(y[test],final[:,1]),'svm':metrics(y[test],final[:,2]),'inception8_mip_fixed299':metrics(y[test],final[:,3]),'four_model_fixed_logit_stack':metrics(y[test],stack),'stack_coef':st['coef'],'stack_intercept':st['intercept']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez(OUT/'test_predictions.npz',uid=np.asarray([rows[j]['uid'] for j in test]),y=y[test],components=final,stack=stack);save_progress('complete',summary=summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True); (OUT/'error.txt').write_text(traceback.format_exc()); raise
