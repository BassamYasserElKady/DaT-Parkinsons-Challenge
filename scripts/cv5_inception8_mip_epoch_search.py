#!/usr/bin/env python3
"""Nested epoch search for scratch Inception-v3 on adaptive-eight-slice MIPs."""
from __future__ import annotations
import copy, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"results/cv5_inception_v3_adaptive8_mip/adaptive8_mips.npy"; LABELS=ROOT/"results/ants_registered_consecutive_resnet18_cbam/split.csv"; OUT=ROOT/"results/cv5_inception8_mip_epoch_search"; SEED=20260931; MAX_EPOCHS=40
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def metric(y,p):
    p=np.clip(p,1e-6,1-1e-6); return {"auroc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p)),"accuracy":float(accuracy_score(y,p>=.5)),"incorrect":int(np.sum((p>=.5)!=y))}
def make_model():
    sys.path.insert(0,str(ROOT/"scripts")); from cv5_inception_v3_adaptive8_mip import make_model as build
    return build()
def prep(x,device): return F.interpolate(torch.from_numpy(x.astype(np.float32,copy=False))[:,None],size=(299,299),mode="bilinear",align_corners=False).to(device)
def predict(m,x,device):
    m.eval(); out=[]
    with torch.inference_mode():
        for s in range(0,len(x),32): out.append(torch.softmax(m(x[s:s+32]),1)[:,1].cpu().numpy())
    return np.concatenate(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True); seed_all(SEED); x=np.load(DATA).astype(np.float32,copy=False); rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); y=np.asarray([int(r["is_pathologic"]) for r in rows],dtype=np.int64); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); oof=np.full(len(y),np.nan); records=[]
    for fold,(outer_train,outer_test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y),1):
        inner_train,inner_val=train_test_split(outer_train,test_size=.2,stratify=y[outer_train],random_state=SEED+fold); seed_all(SEED+fold); m=make_model().to(device); tx,tv,tt=prep(x[inner_train],device),prep(x[inner_val],device),prep(x[outer_test],device); ty=torch.from_numpy(y[inner_train]).to(device); dl=DataLoader(TensorDataset(tx,ty),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(SEED+fold),num_workers=0); c=np.bincount(y[inner_train],minlength=2).astype(float); lossfn=nn.CrossEntropyLoss(weight=torch.tensor(c.sum()/(2*c),dtype=torch.float32,device=device)); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EPOCHS); best_state=None; best_loss=np.inf; best_epoch=0; history=[]
        for ep in range(1,MAX_EPOCHS+1):
            m.train(); total=0.
            for xb,yb in dl:
                if torch.rand(())<.5: xb=xb.flip(-1)
                opt.zero_grad(set_to_none=True); loss=lossfn(m(xb),yb); loss.backward(); opt.step(); total+=float(loss.detach())*len(yb)
            sch.step(); pv=predict(m,tv,device); val=metric(y[inner_val],pv); row={"epoch":ep,"train_loss":total/len(inner_train),"val_log_loss":val["log_loss"],"val_auroc":val["auroc"],"val_accuracy":val["accuracy"]}; history.append(row); print(f"fold={fold} epoch={ep}/{MAX_EPOCHS} train_loss={row['train_loss']:.5f} val_logloss={row['val_log_loss']:.5f} val_auc={row['val_auroc']:.5f}",flush=True)
            if val["log_loss"]<best_loss: best_loss=val["log_loss"]; best_epoch=ep; best_state=copy.deepcopy(m.state_dict())
        m.load_state_dict(best_state); po=predict(m,tt,device); rec={"fold":fold,"inner_train":len(inner_train),"inner_validation":len(inner_val),"outer_test":len(outer_test),"selected_epoch":best_epoch,"inner_best_log_loss":best_loss,"outer":metric(y[outer_test],po)}; records.append(rec); oof[outer_test]=po; (OUT/f"fold{fold}_history.json").write_text(json.dumps(history,indent=2)+"\n"); (OUT/"progress.json").write_text(json.dumps(records,indent=2)+"\n"); print(json.dumps(rec),flush=True); del tx,tv,tt,ty,dl,m
        if device.type=="cuda": torch.cuda.empty_cache()
    summary={"model":"Scratch Inception-v3 adaptive-eight-slice MIP","max_epochs":MAX_EPOCHS,"epoch_selection":"inner-validation minimum log loss","folds":records,"pooled":metric(y,oof)}; np.savez(OUT/"oof_predictions.npz",y=y,prediction=oof); (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
