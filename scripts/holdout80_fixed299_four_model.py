#!/usr/bin/env python3
"""Clean 80/20 evaluation of the repaired four-model stack from raw-registration outputs.

The final 20% holdout is never used for preprocessing statistics, model fitting,
or logistic stack fitting.  The stack is fit from three-fold OOF predictions made
only within the 80% development partition.
"""
from __future__ import annotations

import csv, json, random, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'results/end_to_end_raw_registration_cv5_seed20260931/registered'
LABELS=ROOT/'results/ants_registered_consecutive_resnet18_cbam/split.csv'
PACKAGE=ROOT/'submission_ensemble_five_slice_logit_stack'
BRAIN=PACKAGE/'assets/brain_mask.nii.gz'
OUT=ROOT/'results/holdout80_fixed299_four_model'
SEED=20260931; EPS=1e-6
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(PACKAGE))
from cv5_original_flip_vs_dml import DenseNet121_3D, ResNet18CBAM3D
from src.adaptive_localizer import adaptive_localization
from src.svm_features import extract_features

def seed_all(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def metrics(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return {'auroc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p)),'accuracy':float(accuracy_score(y,p>=.5)),'incorrect':int(np.sum((p>=.5)!=y))}

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p)-np.log1p(-p)

def prep_one(payload):
    i,uid=payload; v=np.asarray(nib.load(str(REG/f'{uid}.nii.gz')).dataobj,dtype=np.float64)
    valid=np.isfinite(v)&(v>0); pos=v[valid]
    if not pos.size or pos.std()<1e-8: raise RuntimeError(f'invalid volume {uid}')
    n=np.zeros(v.shape,np.float32); n[valid]=((v[valid]-pos.mean())/pos.std()).astype(np.float32)
    b,_=adaptive_localization(v); s,e=int(b['start']),int(b['end'])
    five=np.stack([n[:,:,z].T for z in range(s,e+1)],0)
    c=(s+e)//2; mstart=max(0,min(c-4,v.shape[2]-8)); mip=np.max(n[:,:,mstart:mstart+8],2).T
    brain=np.asarray(nib.load(str(BRAIN)).dataobj)>0
    svm,_=extract_features(v,brain)
    return i,five,mip,svm,{'uid':uid,'five_start':s,'five_end':e,'mip_start':mstart,'mip_end':mstart+7}

def build(rows):
    OUT.mkdir(parents=True,exist_ok=True); cache=OUT/'inputs.npz'
    if cache.exists():
        a=np.load(cache); return a['five'],a['mip'],a['svm']
    n=len(rows); five=np.empty((n,5,109,91),np.float32); mip=np.empty((n,109,91),np.float32); svm=np.empty((n,22),float); audit=[None]*n
    with ProcessPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(prep_one,(i,r['uid'])) for i,r in enumerate(rows)]
        for done,f in enumerate(as_completed(fs),1):
            i,a,b,c,d=f.result(); five[i]=a;mip[i]=b;svm[i]=c;audit[i]=d
            if done%100==0 or done==n: print(f'preprocess={done}/{n}',flush=True)
    np.savez(cache,five=five,mip=mip,svm=svm)
    import pandas as pd; pd.DataFrame(audit).to_csv(OUT/'preprocessing_audit.csv',index=False)
    return five,mip,svm

def cnn_data(native,fit):
    means=native[fit].mean((0,2,3),dtype=np.float64).astype(np.float32); stds=native[fit].std((0,2,3),dtype=np.float64).astype(np.float32)
    x=np.empty((len(native),16,64,64),np.float32); zoom=(16/5,64/109,64/91)
    for i in range(len(native)): x[i]=ndimage.zoom((native[i]-means[:,None,None])/stds[:,None,None],zoom,order=1)
    return x

def train3d(model,x,y,fit,batch,epochs,lr,seed,device,name,is_2d=False):
    seed_all(seed); model=model.to(device); counts=np.bincount(y[fit],minlength=2).astype(float); ce=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts),device=device,dtype=torch.float32)); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); tensor=torch.from_numpy(x[fit]); tensor=tensor if is_2d else tensor[:,None]; dl=DataLoader(TensorDataset(tensor,torch.from_numpy(y[fit])),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0,pin_memory=True); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    for ep in range(epochs):
        model.train(); total=0.
        for xb,yb in dl:
            if torch.rand(())<.5: xb=xb.flip(-1)
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): loss=ce(model(xb),yb)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=loss.item()*len(yb)
        sch.step(); print(f'{name} epoch={ep+1}/{epochs} loss={total/len(fit):.5f}',flush=True)
    return model.eval()

def predict3d(m,x,ids,device,is_2d=False):
    tensor=torch.from_numpy(x[ids]); tensor=tensor if is_2d else tensor[:,None]; dl=DataLoader(TensorDataset(tensor),batch_size=4,shuffle=False);out=[]
    with torch.inference_mode():
        for (xb,) in dl: out.append(torch.softmax(m(xb.to(device)),1)[:,1].cpu().numpy())
    return np.concatenate(out)

class Inc(nn.Module):
    def __init__(self):
        super().__init__();self.net=inception_v3(weights=None,aux_logits=False,num_classes=2,init_weights=True); old=self.net.Conv2d_1a_3x3;self.net.Conv2d_1a_3x3.conv=nn.Conv2d(1,old.conv.out_channels,old.conv.kernel_size,stride=old.conv.stride,padding=old.conv.padding,bias=False)
    def forward(self,x): return self.net(x)

def inc_x(mip): return F.interpolate(torch.from_numpy(mip)[:,None],size=(299,299),mode='bilinear',align_corners=False).numpy()

def main():
    seed_all(SEED); OUT.mkdir(parents=True,exist_ok=True); rows=list(csv.DictReader(LABELS.open())); y=np.asarray([int(r['is_pathologic']) for r in rows]); five,mip,svm=build(rows); mx=inc_x(mip); dev=np.sort(next(StratifiedShuffleSplit(1,test_size=.2,random_state=SEED).split(five,y))[0]); test=np.sort(np.setdiff1d(np.arange(len(y)),dev)); np.savez(OUT/'split.npz',development=dev,test=test,y=y)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'device={device} development={len(dev)} test={len(test)}',flush=True)
    # Three-fold OOF within development set for stack fitting.
    oof=np.full((len(y),4),np.nan)
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,y[dev]),1):
        fit,val=dev[a],dev[b]; x=cnn_data(five,fit); print(f'inner_fold={fold} fit={len(fit)} val={len(val)}',flush=True)
        r=train3d(ResNet18CBAM3D(),x,y,fit,8,16,1e-3,SEED+fold,device,f'inner{fold}_resnet');oof[val,0]=predict3d(r,x,val,device);del r
        torch.cuda.empty_cache();d=train3d(DenseNet121_3D(),x,y,fit,2,14,3e-4,SEED+100+fold,device,f'inner{fold}_dense');oof[val,1]=predict3d(d,x,val,device);del d;torch.cuda.empty_cache()
        s=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=SEED+fold));s.fit(svm[fit],y[fit]);oof[val,2]=s.predict_proba(svm[val])[:,1]
        inc=train3d(Inc(),mx,y,fit,16,16,1e-3,SEED+200+fold,device,f'inner{fold}_inception');oof[val,3]=predict3d(inc,mx,val,device);del inc;torch.cuda.empty_cache()
    meta=LogisticRegression(C=1.0,solver='lbfgs',max_iter=5000,random_state=SEED).fit(logit(oof[dev]),y[dev])
    x=cnn_data(five,dev); final=np.empty((len(test),4));r=train3d(ResNet18CBAM3D(),x,y,dev,8,16,1e-3,SEED+500,device,'final_resnet');final[:,0]=predict3d(r,x,test,device);del r;torch.cuda.empty_cache();d=train3d(DenseNet121_3D(),x,y,dev,2,14,3e-4,SEED+600,device,'final_dense');final[:,1]=predict3d(d,x,test,device);del d;torch.cuda.empty_cache();s=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=SEED+700));s.fit(svm[dev],y[dev]);final[:,2]=s.predict_proba(svm[test])[:,1];inc=train3d(Inc(),mx,y,dev,16,16,1e-3,SEED+800,device,'final_inception');final[:,3]=predict3d(inc,mx,test,device);del inc;torch.cuda.empty_cache()
    stack=meta.predict_proba(logit(final))[:,1]; summary={'experiment':'verified raw rigid registration cache -> clean stratified 80/20 split','registration_cache':'results/end_to_end_raw_registration_cv5_seed20260931/registered','test_scan_count':int(len(test)),'development_scan_count':int(len(dev)),'stack_calibration':'three-fold OOF inside development set only','resnet':metrics(y[test],final[:,0]),'densenet':metrics(y[test],final[:,1]),'svm':metrics(y[test],final[:,2]),'inception8_mip_fixed299':metrics(y[test],final[:,3]),'four_model_logit_stack':metrics(y[test],stack),'stack_coef':meta.coef_[0].tolist(),'stack_intercept':float(meta.intercept_[0])};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez(OUT/'test_predictions.npz',uid=np.asarray([rows[i]['uid'] for i in test]),y=y[test],components=final,stack=stack);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
