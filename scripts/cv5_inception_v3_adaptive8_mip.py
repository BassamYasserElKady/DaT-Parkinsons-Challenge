#!/usr/bin/env python3
"""Five-fold scratch Inception-v3 on adaptive eight-slice axial MIPs."""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3

ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/"results/ants_registration_mni"; LABELS=ROOT/"results/ants_registered_consecutive_resnet18_cbam/split.csv"; OUT=ROOT/"results/cv5_inception_v3_adaptive8_mip"; DATA=OUT/"adaptive8_mips.npy"; SEED=20260931; EPOCHS=16
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def metric(y,p):
    p=np.clip(p,1e-6,1-1e-6); return {"auroc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p)),"accuracy":float(accuracy_score(y,p>=.5)),"incorrect":int(np.sum((p>=.5)!=y))}
def build_data(rows):
    if DATA.exists(): return np.load(DATA).astype(np.float32,copy=False)
    sys.path.insert(0,str(ROOT/"submission_ensemble_sigma075_8slice_svm/src")); from adaptive_localizer import adaptive_localization
    OUT.mkdir(parents=True,exist_ok=True); data=np.empty((len(rows),109,91),dtype=np.float32); audit=[]
    for i,row in enumerate(rows):
        vol=np.asarray(nib.load(str(REG/f"{row['uid']}.nii.gz")).dataobj,dtype=np.float32); best,_=adaptive_localization(vol); center=(int(best["start"])+int(best["end"]))//2; start=max(0,min(center-4,vol.shape[2]-8)); end=start+8; valid=np.isfinite(vol)&(vol>0); pos=vol[valid]; norm=np.zeros_like(vol); norm[valid]=(vol[valid]-float(pos.mean()))/max(float(pos.std()),1e-8); data[i]=np.max(norm[:,:,start:end],axis=2).T; audit.append({"uid":row["uid"],"start":start,"end":end-1,"localizer_start":int(best["start"]),"localizer_end":int(best["end"])})
        if (i+1)%100==0 or i+1==len(rows): print(f"prepared={i+1}/{len(rows)}",flush=True)
    np.save(DATA,data); 
    with (OUT/"slice_selection.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)
    return data
def make_model():
    m=inception_v3(weights=None,aux_logits=False,num_classes=2,init_weights=True); old=m.Conv2d_1a_3x3; old.conv=nn.Conv2d(1,old.conv.out_channels,old.conv.kernel_size,stride=old.conv.stride,padding=old.conv.padding,bias=False); return m
def prep(x,device): return F.interpolate(torch.from_numpy(x.astype(np.float32,copy=False))[:,None],size=(299,299),mode="bilinear",align_corners=False).to(device)
def train_fold(x,y,tr,te,fold,device):
    seed_all(SEED+fold); m=make_model().to(device); c=np.bincount(y[tr],minlength=2).astype(float); lossfn=nn.CrossEntropyLoss(weight=torch.tensor(c.sum()/(2*c),dtype=torch.float32,device=device)); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); tx,ex=prep(x[tr],device),prep(x[te],device); ty=torch.from_numpy(y[tr]).to(device); dl=DataLoader(TensorDataset(tx,ty),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(SEED+fold),num_workers=0)
    for ep in range(1,EPOCHS+1):
        m.train(); total=0.
        for xb,yb in dl:
            if torch.rand(())<.5: xb=xb.flip(-1)
            opt.zero_grad(set_to_none=True); l=lossfn(m(xb),yb); l.backward(); opt.step(); total+=float(l.detach())*len(yb)
        sch.step(); print(f"fold={fold} epoch={ep}/{EPOCHS} loss={total/len(tr):.6f} lr={opt.param_groups[0]['lr']:.8g}",flush=True)
    m.eval(); out=[]
    with torch.inference_mode():
        for s in range(0,len(te),32): out.append(torch.softmax(m(ex[s:s+32]),1)[:,1].cpu().numpy())
    del tx,ex,ty,dl,m
    if device.type=="cuda": torch.cuda.empty_cache()
    return np.concatenate(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True); seed_all(SEED); rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); y=np.asarray([int(r["is_pathologic"]) for r in rows],dtype=np.int64); x=build_data(rows); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); oof=np.full(len(y),np.nan); records=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y),1):
        p=train_fold(x,y,tr,te,fold,device); oof[te]=p; rec={"fold":fold,"train":int(len(tr)),"test":int(len(te)),**metric(y[te],p)}; records.append(rec); (OUT/"progress.json").write_text(json.dumps(records,indent=2)+"\n"); print(json.dumps(rec),flush=True)
    summary={"model":"Inception-v3 scratch, adaptive-eight-slice axial MIP","weights":"None (not ImageNet)","cbam":False,"epochs":EPOCHS,"folds":records,"pooled":metric(y,oof)}; np.savez(OUT/"oof_predictions.npz",y=y,prediction=oof); (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__": main()
