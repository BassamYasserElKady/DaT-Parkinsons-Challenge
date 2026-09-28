#!/usr/bin/env python3
"""Train the final scratch Inception-v3 on all adaptive-eight-slice MIPs."""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"results/cv5_inception_v3_adaptive8_mip/adaptive8_mips.npy"; LABELS=ROOT/"results/ants_registered_consecutive_resnet18_cbam/split.csv"; OUT=ROOT/"results/full_inception8_mip_submission"; SEED=20260931; EPOCHS=16
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def main():
    seed_all(SEED); sys.path.insert(0,str(ROOT/"scripts")); from cv5_inception_v3_adaptive8_mip import make_model
    OUT.mkdir(parents=True,exist_ok=True); x=np.load(DATA).astype(np.float32,copy=False); rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); y=np.asarray([int(r["is_pathologic"]) for r in rows],dtype=np.int64); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x.shape!=(1362,109,91): raise RuntimeError(f"unexpected MIP shape {x.shape}")
    # Match cv5_inception_v3_adaptive8_mip.py exactly: the model is trained on
    # bilinearly resized 299 x 299 MIPs, not the native 109 x 91 array.
    x_tensor=F.interpolate(torch.from_numpy(x)[:,None],size=(299,299),mode="bilinear",align_corners=False)
    model=make_model().to(device); loader=DataLoader(TensorDataset(x_tensor,torch.from_numpy(y)),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(SEED),num_workers=0); counts=np.bincount(y,minlength=2).astype(float); criterion=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); hist=[]
    for ep in range(1,EPOCHS+1):
        model.train(); total=0.
        for xb,yb in loader:
            if torch.rand(())<.5: xb=xb.flip(-1)
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); loss=criterion(model(xb),yb); loss.backward(); opt.step(); total+=float(loss.detach())*len(yb)
        sch.step(); row={"epoch":ep,"loss":total/len(y),"learning_rate":opt.param_groups[0]["lr"]}; hist.append(row); print(f"epoch={ep}/{EPOCHS} loss={row['loss']:.6f} lr={row['learning_rate']:.8g}",flush=True)
    torch.save({"inception_state_dict":model.cpu().state_dict()},OUT/"inception8_mip_weights.pt"); (OUT/"training_metadata.json").write_text(json.dumps({"training_scan_count":1362,"epochs":EPOCHS,"optimizer":"AdamW","scheduler":"CosineAnnealingLR","augmentation":"random_horizontal_flip_only","input":"adaptive eight-slice axial MIP resized bilinearly to 299x299","weights":"None (not ImageNet)"},indent=2)+"\n")
    with (OUT/"history.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
if __name__=="__main__": main()
