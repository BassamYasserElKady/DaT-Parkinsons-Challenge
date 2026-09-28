#!/usr/bin/env python3
"""Five-fold scratch Inception-v3 plus CBAM on adaptive-five-slice MIPs."""
from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3

ROOT=Path(__file__).resolve().parents[1]; MIPS=ROOT/"results/adaptive5_mip_2d/adaptive5_mips.npy"; LABELS=ROOT/"results/ants_registered_consecutive_resnet18_cbam/split.csv"; OUT=ROOT/"results/cv5_inception_v3_cbam_adaptive5_mip"; SEED=20260931; EPOCHS=16

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def metric(y,p):
    p=np.clip(p,1e-6,1-1e-6); return {"auroc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p)),"accuracy":float(accuracy_score(y,p>=.5)),"incorrect":int(np.sum((p>=.5)!=y))}

class CBAM2D(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__(); hidden=max(channels//reduction,4); self.mlp=nn.Sequential(nn.Conv2d(channels,hidden,1,bias=False),nn.ReLU(inplace=True),nn.Conv2d(hidden,channels,1,bias=False)); self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        avg=self.mlp(F.adaptive_avg_pool2d(x,1)); mx=self.mlp(F.adaptive_max_pool2d(x,1)); x=x*torch.sigmoid(avg+mx); s=torch.cat((x.mean(1,keepdim=True),x.amax(1,keepdim=True)),1); return x*torch.sigmoid(self.spatial(s))

def make_model():
    m=inception_v3(weights=None,aux_logits=False,num_classes=2,init_weights=True); old=m.Conv2d_1a_3x3; old.conv=nn.Conv2d(1,old.conv.out_channels,old.conv.kernel_size,stride=old.conv.stride,padding=old.conv.padding,bias=False)
    m.Mixed_5d=nn.Sequential(m.Mixed_5d,CBAM2D(288)); m.Mixed_6e=nn.Sequential(m.Mixed_6e,CBAM2D(768)); m.Mixed_7c=nn.Sequential(m.Mixed_7c,CBAM2D(2048)); return m

def prepare(x,device): return F.interpolate(torch.from_numpy(x.astype(np.float32,copy=False))[:,None],size=(299,299),mode="bilinear",align_corners=False).to(device)

def train_fold(x,y,tr,te,fold,device):
    seed_all(SEED+fold); model=make_model().to(device); counts=np.bincount(y[tr],minlength=2).astype(float); criterion=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); tx,ex=prepare(x[tr],device),prepare(x[te],device); ty=torch.from_numpy(y[tr]).to(device); dl=DataLoader(TensorDataset(tx,ty),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(SEED+fold),num_workers=0)
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0.0
        for xb,yb in dl:
            if torch.rand(())<.5: xb=xb.flip(-1)
            opt.zero_grad(set_to_none=True); loss=criterion(model(xb),yb); loss.backward(); opt.step(); total+=float(loss.detach())*len(yb)
        sched.step(); print(f"fold={fold} epoch={epoch}/{EPOCHS} loss={total/len(tr):.6f} lr={opt.param_groups[0]['lr']:.8g}",flush=True)
    model.eval(); out=[]
    with torch.inference_mode():
        for start in range(0,len(te),32): out.append(torch.softmax(model(ex[start:start+32]),1)[:,1].cpu().numpy())
    del tx,ex,ty,dl,model
    if device.type=="cuda": torch.cuda.empty_cache()
    return np.concatenate(out)

def main():
    OUT.mkdir(parents=True,exist_ok=True); seed_all(SEED); x=np.load(MIPS).astype(np.float32,copy=False); rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); y=np.asarray([int(r["is_pathologic"]) for r in rows],dtype=np.int64)
    if x.shape!=(1362,109,91) or len(y)!=len(x): raise RuntimeError(f"unexpected shapes {x.shape} {len(y)}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); oof=np.full(len(y),np.nan); records=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y),1):
        p=train_fold(x,y,tr,te,fold,device); oof[te]=p; rec={"fold":fold,"train":int(len(tr)),"test":int(len(te)),**metric(y[te],p)}; records.append(rec); (OUT/"progress.json").write_text(json.dumps(records,indent=2)+"\n"); print(json.dumps(rec),flush=True)
    summary={"model":"Inception-v3 scratch + CBAM, grayscale adaptive-five-slice axial MIP","weights":"None (not ImageNet)","attention_placement":["Mixed_5d","Mixed_6e","Mixed_7c/final bottleneck"],"epochs":EPOCHS,"input_shape":[1,299,299],"folds":records,"pooled":metric(y,oof)}; np.savez(OUT/"oof_predictions.npz",y=y,prediction=oof); (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
