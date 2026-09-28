#!/usr/bin/env python3
from __future__ import annotations
import copy, csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/resnet18_sigma075_flip_25epochs'
ROWS=ROOT/'results/ants_registered_consecutive_resnet18_cbam/split.csv'
NATIVE=ROOT/'results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy'
SEED=20260931

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(((p>=.5)!=y).sum())}

def predict(model,x,ids,device):
    dl=DataLoader(TensorDataset(torch.from_numpy(x[ids]).unsqueeze(1)),batch_size=8,shuffle=False)
    out=[];model.eval()
    with torch.inference_mode():
        for (xb,) in dl:out.extend(torch.softmax(model(xb.to(device)),1)[:,1].cpu().tolist())
    return np.asarray(out)

def main():
    OUT.mkdir(parents=True,exist_ok=True);seed_all(SEED)
    rows=list(csv.DictReader(ROWS.open()));y=np.asarray([int(r['is_pathologic']) for r in rows],dtype=np.int64)
    ids={name:np.asarray([i for i,r in enumerate(rows) if r['split']==name]) for name in ('train','validation','test')}
    native=np.load(NATIVE,mmap_mode='r');tr=ids['train']
    means=np.asarray(native[tr]).mean(axis=(0,2,3),dtype=np.float64).astype(np.float32)
    stds=np.asarray(native[tr]).std(axis=(0,2,3),dtype=np.float64).astype(np.float32)
    x=np.empty((len(y),16,64,64),dtype=np.float32)
    for i,v in enumerate(native):
        z=(np.asarray(v)-means[:,None,None])/stds[:,None,None]
        x[i]=ndimage.zoom(z,(2,64/109,64/91),order=1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'device={device} train={len(tr)} validation={len(ids["validation"])} test={len(ids["test"])}',flush=True)
    model=ResNet18CBAM3D().to(device);counts=np.bincount(y[tr],minlength=2).astype(float)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32,device=device))
    dl=DataLoader(TensorDataset(torch.from_numpy(x[tr]).unsqueeze(1),torch.from_numpy(y[tr])),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(SEED),num_workers=0,pin_memory=device.type=='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');history=[];best=None;best_loss=float('inf')
    for epoch in range(1,26):
        model.train();total=0
        for xb,yb in dl:
            if torch.rand(())<.5:xb=xb.flip(-1)
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):loss=loss_fn(model(xb),yb)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=float(loss.detach())*len(yb)
        sched.step();vp=predict(model,x,ids['validation'],device);vm=metrics(y[ids['validation']],vp)
        row={'epoch':epoch,'train_loss':total/len(tr),'learning_rate':opt.param_groups[0]['lr'],**{f'val_{k}':v for k,v in vm.items()}};history.append(row)
        if vm['log_loss']<best_loss:best_loss=vm['log_loss'];best={'epoch':epoch,'state':copy.deepcopy(model.cpu().state_dict()),'validation':vm};model.to(device)
        with (OUT/'history.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        print(json.dumps(row),flush=True)
    model.load_state_dict(best['state']);model.to(device);tp=predict(model,x,ids['test'],device);tm=metrics(y[ids['test']],tp)
    summary={'selection_metric':'validation log_loss','best_epoch':best['epoch'],'validation':best['validation'],'test':tm,'settings':{'epochs_max':25,'scheduler':'CosineAnnealingLR T_max=25','augmentation':'horizontal flip only','sigma':0.75,'input':'8 slices interpolated to 16','lr':1e-3,'weight_decay':1e-4}}
    torch.save({'model_state_dict':best['state'],**summary},OUT/'best_model.pt');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'test_predictions.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=['uid','actual','probability']);w.writeheader();w.writerows({'uid':rows[i]['uid'],'actual':int(y[i]),'probability':float(p)} for i,p in zip(ids['test'],tp))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
