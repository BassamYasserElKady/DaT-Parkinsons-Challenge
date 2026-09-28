#!/usr/bin/env python3
"""Clean 80/20 CNN: MIP-guided shapes with four + sixteen regions per side."""
from __future__ import annotations

import csv, json, random, traceback
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
SHAPES=ROOT/'results/adaptive8_mip6_guided_hierarchical_quarters/labels'
MAPS=ROOT/'results/adaptive8_mip6_guided_sliding_intensity_pyramid/maps'
LABELS=ROOT/'results/ants_registered_consecutive_resnet18_cbam/split.csv'
OUT=ROOT/'results/holdout80_mip6_guided_hierarchical_shape_intensity_cnn'
SEED=20260931; EPOCHS=16; EPS=1e-6

def seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def metric(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}

def make_input(uid):
    labels=np.asarray(nib.load(str(SHAPES/f'{uid}_adaptive8_hierarchical_quarters.nii.gz')).dataobj,dtype=np.uint8)
    channels=np.zeros((48,*labels.shape),np.float32)
    # Full shapes: union their 16 second-level child labels.
    channels[0]=(labels>=9)&(labels<=24); channels[1]=(labels>=25)&(labels<=40)
    # Four parent quarters, reconstructed as their four children.
    for q in range(4):
        channels[2+q]=(labels>=9+4*q)&(labels<=12+4*q)
        channels[6+q]=(labels>=25+4*q)&(labels<=28+4*q)
    # The 16 child regions on each side.
    for child in range(16):
        channels[10+child]=labels==9+child
        channels[26+child]=labels==25+child
    sparse=np.load(MAPS/f'{uid}_adaptive8_sliding_pyramid.npz'); xyz,side=sparse['xyz'],sparse['side']; raw=sparse['intensity_1'].astype(np.float32); pos=raw[raw>0]; scale=max(float(np.median(pos)) if len(pos) else 1.,1e-6)
    for level,key in enumerate(('intensity_1','intensity_2','intensity_3')):
        value=sparse[key].astype(np.float32)/scale
        for s,channel in ((0,42+level),(1,45+level)):
            select=side==s
            if np.any(select):
                c=xyz[select]; channels[channel,c[:,0],c[:,1],c[:,2]]=value[select]
    tensor=torch.from_numpy(channels.transpose(0,3,2,1)).unsqueeze(0)
    return F.interpolate(tensor,size=(8,64,64),mode='trilinear',align_corners=False).squeeze(0).numpy().astype(np.float16)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.f=nn.Sequential(nn.Conv3d(48,40,3,padding=1,bias=False),nn.BatchNorm3d(40),nn.GELU(),nn.Conv3d(40,40,3,padding=1,bias=False),nn.BatchNorm3d(40),nn.GELU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(40,72,3,padding=1,bias=False),nn.BatchNorm3d(72),nn.GELU(),nn.MaxPool3d((2,2,2)),nn.Conv3d(72,96,3,padding=1,bias=False),nn.BatchNorm3d(96),nn.GELU(),nn.AdaptiveAvgPool3d(1)); self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.40),nn.Linear(96,2))
    def forward(self,x): return self.h(self.f(x))

def main():
    seed(SEED); OUT.mkdir(parents=True,exist_ok=True); rows=list(csv.DictReader(LABELS.open())); uids=[r['uid'] for r in rows]; y=np.asarray([int(r['is_pathologic']) for r in rows],np.int64)
    development,test=next(StratifiedShuffleSplit(1,test_size=.2,random_state=SEED).split(np.zeros(len(y)),y)); development,test=np.sort(development),np.sort(test)
    x=np.empty((len(y),48,8,64,64),np.float16)
    for i,uid in enumerate(uids,1):
        x[i-1]=make_input(uid)
        if i%100==0 or i==len(y): print(f'inputs={i}/{len(y)}',flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=Net().to(device); counts=np.bincount(y[development],minlength=2).astype(float); ce=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts),device=device,dtype=torch.float32)); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    train=DataLoader(TensorDataset(torch.from_numpy(x[development]),torch.from_numpy(y[development])),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    for e in range(1,EPOCHS+1):
        model.train(); total=0.
        for xb,yb in train:
            xb,yb=xb.to(device,dtype=torch.float32),yb.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): loss=ce(model(xb),yb)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=float(loss.detach())*len(yb)
        sch.step();print(f'epoch={e}/{EPOCHS} loss={total/len(development):.6f}',flush=True)
    model.eval(); prediction=[]
    with torch.inference_mode():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(x[test])),batch_size=16): prediction.append(torch.softmax(model(xb.to(device,dtype=torch.float32)),1)[:,1].cpu().numpy())
    p=np.concatenate(prediction); summary={'experiment':'clean stratified 80/20 holdout; MIP-6-guided contour shapes','input':'48 channels: two full shapes, eight first-level quarters, 32 second-level subquarters, six intensity maps','development_scans':int(len(development)),'test_scans':int(len(test)),'epochs':EPOCHS,'test':metric(y[test],p)}
    np.savez_compressed(OUT/'test_predictions.npz',uid=np.asarray([uids[i] for i in test]),y=y[test],probability=p);torch.save({'state_dict':model.cpu().state_dict()},OUT/'model.pt');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    try: main()
    except Exception: OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.txt').write_text(traceback.format_exc());raise
