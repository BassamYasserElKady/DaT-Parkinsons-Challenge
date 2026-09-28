#!/usr/bin/env python3
"""Apparent full-cohort evaluation of the freshly trained submission ensemble."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import ants,numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'submission_ensemble_four_model_logit_stack'
REG=ROOT/'results/end_to_end_raw_registration_cv5_seed20260931/registered'
LABELS=ROOT/'results/ants_registered_consecutive_resnet18_cbam/split.csv'
OUT=ROOT/'results/fresh_full1362_ensemble_apparent'
sys.path.insert(0,str(PACKAGE))
from src.model import load_models
from src.preprocessing import prepare_tensor_from_registered,prepare_inception_mip
from src.svm_features import extract_features
from src.svm_model import RbfSvm

def metric(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p)),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}

def main():
    args=argparse.ArgumentParser();args.add_argument('--n',type=int,default=0);opt=args.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); rows=list(csv.DictReader(LABELS.open())); y=np.asarray([int(r['is_pathologic']) for r in rows])
    if opt.n:
        ids=next(StratifiedShuffleSplit(n_splits=1,train_size=opt.n,random_state=20260931).split(np.zeros(len(y)),y))[0]; rows=[rows[i] for i in ids]; y=y[ids]
    std=json.loads((PACKAGE/'assets/standardization.json').read_text());md=json.loads((PACKAGE/'assets/model_metadata.json').read_text());dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');cnn,inc=load_models(PACKAGE/'assets/model/full_ensemble_weights.pt',PACKAGE/'assets/model/inception8_mip_weights.pt',dev);brain=ants.image_read(str(PACKAGE/'assets/brain_mask.nii.gz')).numpy()>0;svm=RbfSvm(PACKAGE/'assets/model/morphometric_intensity_rbf_svm.npz');p=np.empty((len(rows),4),float)
    for start in range(0,len(rows),8):
        batch=rows[start:start+8]; t3=[];t2=[];sf=[]
        for row in batch:
            v=np.asarray(ants.image_read(str(REG/f"{row['uid']}.nii.gz")).numpy(),dtype=np.float64);t3.append(prepare_tensor_from_registered(v,std['channel_means'],std['channel_stds']).squeeze(0));t2.append(prepare_inception_mip(v).squeeze(0));sf.append(svm.predict_probability(extract_features(v,brain)[0]))
        with torch.inference_mode(): r,d=cnn(torch.stack(t3).to(dev)); i=inc(torch.stack(t2).to(dev))
        p[start:start+len(batch)]=np.column_stack((r.cpu().numpy(),d.cpu().numpy(),sf,i.cpu().numpy()))
        if start%80==0 or start+len(batch)==len(rows): print(f'inference={start+len(batch)}/{len(rows)}',flush=True)
    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));st=md['logit_stack'];ensemble=1/(1+np.exp(-np.clip(z@np.asarray(st['coef'])+st['intercept'],-40,40)));summary={'evaluation':'apparent training-set performance; all scans used to fit final base-model weights','scan_count':len(rows),'resnet':metric(y,p[:,0]),'densenet':metric(y,p[:,1]),'svm':metric(y,p[:,2]),'inception8_mip':metric(y,p[:,3]),'four_model_stack':metric(y,ensemble)};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');pd.DataFrame({'uid':[r['uid'] for r in rows],'actual':y,'resnet':p[:,0],'densenet':p[:,1],'svm':p[:,2],'inception':p[:,3],'ensemble':ensemble}).to_csv(OUT/'predictions.csv',index=False);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
