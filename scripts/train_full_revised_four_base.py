#!/usr/bin/env python3
"""Train full-cohort CNNs, refit the RBF SVM, and save stack parameters."""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
FEATURES = ROOT / "results/deployable_three_model_svm/features.npy"
FEATURE_AUDIT = ROOT / "results/deployable_three_model_svm/window_audit.csv"
OUT = ROOT / "results/full_revised_four_model_submission"
SEED = 20260931

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)

def train_model(name, model, x, y, batch, epochs, lr, device):
    seed_all(SEED + (1 if name == "resnet18" else 2)); model = model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)), batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0, pin_memory=device.type == "cuda")
    counts = np.bincount(y, minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(2*counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); history=[]
    for epoch in range(1, epochs + 1):
        model.train(); total=0.0
        for xb, yb in loader:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            xb,yb=xb.to(device, non_blocking=True),yb.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"): loss=criterion(model(xb),yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += float(loss.detach())*len(yb)
        scheduler.step(); row={"epoch":epoch,"loss":total/len(y),"learning_rate":optimizer.param_groups[0]["lr"]}; history.append(row)
        print(f"{name} epoch={epoch}/{epochs} loss={row['loss']:.6f} lr={row['learning_rate']:.8g}",flush=True)
    return model.cpu().state_dict(), history

def load_oof(rows):
    labels={r["uid"]:int(r["is_pathologic"]) for r in rows}; values={}
    for fold in range(1,6):
        a=np.load(ROOT/f"results/fold{fold}_five_slice_no_gaussian_three_model_nested/predictions.npz")
        for uid,y,r,d,s in zip(a["uid"],a["actual"],a["resnet"],a["densenet"],a["svm"]):
            uid=str(uid)
            if uid in values or labels.get(uid)!=int(y): raise RuntimeError(f"bad OOF record: {uid}")
            values[uid]=[float(r),float(d),float(s)]
    if set(values)!=set(labels): raise RuntimeError(f"OOF coverage {len(values)}/{len(labels)}")
    return np.asarray([values[r["uid"]] for r in rows],dtype=np.float64)

def fit_svm(rows):
    features=np.load(FEATURES).astype(np.float64,copy=False); audit=list(csv.DictReader(FEATURE_AUDIT.open(newline="")))
    labels={r["uid"]:int(r["is_pathologic"]) for r in rows}; y=np.asarray([labels[r["uid"]] for r in audit],dtype=np.int64)
    if len(audit)!=len(features): raise RuntimeError("SVM feature/audit mismatch")
    pipe=make_pipeline(StandardScaler(),SVC(C=1,kernel="rbf",gamma="scale",probability=True,random_state=SEED)); pipe.fit(features,y)
    sc=pipe.named_steps["standardscaler"]; sv=pipe.named_steps["svc"]
    arrays={"mean":sc.mean_,"scale":sc.scale_,"support_vectors":sv.support_vectors_,"dual_coef":sv.dual_coef_[0],"intercept":sv.intercept_[0],"gamma":sv._gamma,"prob_a":sv.probA_[0],"prob_b":sv.probB_[0]}
    joblib.dump(pipe,OUT/"morphometric_intensity_rbf_svm.joblib",compress=3); np.savez(OUT/"morphometric_intensity_rbf_svm.npz",**arrays)
    return arrays

def main():
    OUT.mkdir(parents=True,exist_ok=True); seed_all(SEED); sys.path.insert(0,str(ROOT/"scripts"))
    from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D
    from train_registered_consecutive_densenet121_3d import DenseNet121_3D
    rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); y=np.asarray([int(r["is_pathologic"]) for r in rows],dtype=np.int64); x=np.load(INPUTS).astype(np.float32,copy=False)
    if x.shape!=(1362,16,64,64) or len(y)!=1362: raise RuntimeError(f"unexpected cohort shape {x.shape}, labels {len(y)}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device} scans={len(y)}",flush=True)
    rs,rh=train_model("resnet18",ResNet18CBAM3D(),x,y,8,16,1e-3,device)
    if device.type=="cuda": torch.cuda.empty_cache()
    ds,dh=train_model("densenet121",DenseNet121_3D(),x,y,2,14,3e-4,device)
    svm_arrays=fit_svm(rows); oof=load_oof(rows); base=np.asarray([.40,.15,.45]); meta=LogisticRegression(C=1.0,solver="lbfgs",max_iter=5000,random_state=SEED).fit(logit(oof)*base[None,:],y)
    torch.save({"resnet_state_dict":rs,"densenet_state_dict":ds},OUT/"full_ensemble_weights.pt")
    metadata={"training_scan_count":1362,"input_shape":[1,16,64,64],"resnet_epochs":16,"densenet_epochs":14,"scheduler":"CosineAnnealingLR","augmentation":"random_horizontal_flip_only","post_registration_gaussian_sigma":0.0,"adaptive_localizer_slices":5,"logit_stack":{"base_weights":{"resnet":.40,"densenet":.15,"svm":.45},"coef":meta.coef_[0].tolist(),"intercept":float(meta.intercept_[0]),"oof_source":"five-fold leakage-controlled nested predictions"}}
    (OUT/"training_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n"); np.savez(OUT/"logit_stack_oof.npz",probabilities=oof,labels=y)
    for name,hist in (("resnet_history.csv",rh),("densenet_history.csv",dh)):
        with (OUT/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=="__main__": main()
