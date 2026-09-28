#!/usr/bin/env python3
from __future__ import annotations
import csv, json, random
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_resnet_sigma075_occipital"
NATIVE = ROOT / "results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
REGISTERED = ROOT / "results/ants_registration_mni"
OCC_MASK = ROOT / "assets/atlases/fsl/masks/occipital_cortex.nii.gz"
SEED = 20260931

class ResNetOccipital(nn.Module):
    def __init__(self):
        super().__init__()
        base = ResNet18CBAM3D()
        self.stem, self.layer1, self.layer2 = base.stem, base.layer1, base.layer2
        self.layer3, self.layer4, self.pool, self.dropout = base.layer3, base.layer4, base.pool, base.dropout
        self.fc = nn.Linear(513, 2)

    def forward(self, image, occipital):
        x = self.stem(image); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.dropout(self.pool(x).flatten(1))
        return self.fc(torch.cat([x, occipital[:, None]], dim=1))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def score(y, p):
    return {"auroc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"accuracy":float(accuracy_score(y,p>=.5)),"incorrect":int(((p>=.5)!=y).sum())}

def main():
    OUT.mkdir(parents=True, exist_ok=True); seed_all(SEED)
    rows=list(csv.DictReader(ROWS.open())); y=np.array([int(r["is_pathologic"]) for r in rows],dtype=np.int64)
    native=np.load(NATIVE,mmap_mode="r"); mask=nib.load(str(OCC_MASK)).get_fdata()>0
    occ=[]
    for i,r in enumerate(rows,1):
        a=nib.load(str(REGISTERED/f'{r["uid"]}.nii.gz')).get_fdata(dtype=np.float32)
        occ.append(float(np.mean(a[mask])))
    occ=np.log1p(np.maximum(np.asarray(occ,dtype=np.float32),0))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device}",flush=True)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y)); oof=np.full(len(y),np.nan); details=[]
    for fold,(tr,va) in enumerate(splits,1):
        means=np.asarray(native[tr]).mean(axis=(0,2,3),dtype=np.float64).astype(np.float32)
        stds=np.asarray(native[tr]).std(axis=(0,2,3),dtype=np.float64).astype(np.float32)
        x=np.empty((len(y),16,64,64),dtype=np.float32)
        for i,v in enumerate(native):
            z=(np.asarray(v)-means[:,None,None])/stds[:,None,None]
            x[i]=ndimage.zoom(z,(2,64/109,64/91),order=1)
        om,os=float(occ[tr].mean()),float(occ[tr].std()); oz=((occ-om)/max(os,1e-8)).astype(np.float32)
        model=ResNetOccipital().to(device); counts=np.bincount(y[tr],minlength=2).astype(float)
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32,device=device))
        loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]).unsqueeze(1),torch.from_numpy(oz[tr]),torch.from_numpy(y[tr])),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(SEED+fold),num_workers=0,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=16)
        scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
        for epoch in range(1,17):
            model.train(); total=0
            for xb,ob,yb in loader:
                if torch.rand(())<.5: xb=xb.flip(-1)
                xb,ob,yb=xb.to(device),ob.to(device),yb.to(device); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=="cuda"): loss=criterion(model(xb,ob),yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=float(loss.detach())*len(yb)
            sched.step(); print(f"fold={fold}/5 epoch={epoch}/16 loss={total/len(tr):.6f}",flush=True)
        model.eval(); pp=[]
        dl=DataLoader(TensorDataset(torch.from_numpy(x[va]).unsqueeze(1),torch.from_numpy(oz[va])),batch_size=8,shuffle=False)
        with torch.inference_mode():
            for xb,ob in dl: pp.extend(torch.softmax(model(xb.to(device),ob.to(device)),1)[:,1].cpu().tolist())
        oof[va]=pp; fm={"fold":fold,**score(y[va],oof[va])};details.append(fm);print(json.dumps(fm),flush=True)
        del model,x; torch.cuda.empty_cache()
        np.savez(OUT/"oof_progress.npz",probability=oof)
        (OUT/"progress.json").write_text(json.dumps(details,indent=2)+"\n")
    summary={"experiment":"ResNet-18 CBAM + log1p occipital mean","oof":score(y,oof),"baseline_resnet":{"auroc":0.9557318705717178,"log_loss":0.2653448560596244,"accuracy":0.8876651982378855},"folds":details}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    pd_rows=[{"uid":r["uid"],"actual":int(y[i]),"occipital_mean_log1p":float(occ[i]),"oof_probability":float(oof[i])} for i,r in enumerate(rows)]
    with (OUT/"oof_predictions.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(pd_rows[0]));w.writeheader();w.writerows(pd_rows)
    print(json.dumps(summary["oof"],indent=2),flush=True)

if __name__=="__main__": main()
