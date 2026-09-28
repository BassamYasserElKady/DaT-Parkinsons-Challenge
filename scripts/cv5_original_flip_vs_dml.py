#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cv5_original_flip_vs_dml"
SEED = 20260931
TEMP = 2.0
KL_WEIGHT = 0.5


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def loader(x, y, ids, batch, shuffle, seed):
    return DataLoader(TensorDataset(torch.from_numpy(x[ids]).unsqueeze(1), torch.from_numpy(y[ids])), batch_size=batch, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True)


def predict(model, x, ids, device):
    dl = DataLoader(TensorDataset(torch.from_numpy(x[ids]).unsqueeze(1)), batch_size=4, shuffle=False, pin_memory=True)
    out = []
    model.eval()
    with torch.inference_mode():
        for (xb,) in dl:
            out.append(torch.softmax(model(xb.to(device, non_blocking=True)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def class_loss(y, ids, device):
    counts = np.bincount(y[ids], minlength=2).astype(float)
    return nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))


def train_independent(model, x, y, ids, batch, epochs, lr, device, seed, name):
    seed_all(seed); model = model.to(device)
    dl = loader(x, y, ids, batch, True, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = class_loss(y, ids, device)
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in dl:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = ce(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += float(loss.detach()) * len(yb)
        sched.step()
        print(f"{name} epoch={epoch}/{epochs} loss={total / len(ids):.6f}", flush=True)
    return model.eval()


def train_dml(x, y, ids, device, seed, fold):
    seed_all(seed)
    r, d = ResNet18CBAM3D().to(device), DenseNet121_3D().to(device)
    dl = loader(x, y, ids, 2, True, seed)
    ce = class_loss(y, ids, device)
    ropt = torch.optim.AdamW(r.parameters(), lr=1e-3, weight_decay=1e-4)
    dopt = torch.optim.AdamW(d.parameters(), lr=3e-4, weight_decay=1e-4)
    rs = torch.optim.lr_scheduler.CosineAnnealingLR(ropt, T_max=16); ds = torch.optim.lr_scheduler.CosineAnnealingLR(dopt, T_max=16)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, 17):
        r.train(); d.train(); total = 0.0
        for xb, yb in dl:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            ropt.zero_grad(set_to_none=True); dopt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                rl, dlgt = r(xb), d(xb)
                rp = F.log_softmax(rl / TEMP, 1); dp = F.softmax(dlgt / TEMP, 1)
                dlog = F.log_softmax(dlgt / TEMP, 1); rprob = F.softmax(rl / TEMP, 1)
                kl = (F.kl_div(rp, dp, reduction="batchmean") + F.kl_div(dlog, rprob, reduction="batchmean")) * TEMP * TEMP / 2
                loss = ce(rl, yb) + ce(dlgt, yb) + KL_WEIGHT * kl
            scaler.scale(loss).backward(); scaler.step(ropt); scaler.step(dopt); scaler.update(); total += float(loss.detach()) * len(yb)
        rs.step(); ds.step()
        print(f"dml fold={fold} epoch={epoch}/16 loss={total / len(ids):.6f}", flush=True)
    return r.eval(), d.eval()


def bounded_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def metric(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "accuracy": float(accuracy_score(y, p >= 0.5))}


def calibrated(val_y, val_p, outer_p):
    c = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=SEED).fit(bounded_logit(val_p).reshape(-1, 1), val_y)
    return c.predict_proba(bounded_logit(outer_p).reshape(-1, 1))[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True); seed_all(SEED)
    rows = list(csv.DictReader((ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open()))
    y = np.asarray([int(r["is_pathologic"]) for r in rows], dtype=np.int64)
    x = np.load(ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy").astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    all_rows = []; folds = []
    for fold, (outer_train, outer) in enumerate(skf.split(x, y), 1):
        train, val = train_test_split(outer_train, test_size=0.2, stratify=y[outer_train], random_state=SEED + fold)
        print(f"fold={fold}/5 train={len(train)} val={len(val)} outer={len(outer)}", flush=True)
        r = train_independent(ResNet18CBAM3D(), x, y, train, 8, 16, 1e-3, device, SEED + fold, f"original_r18_f{fold}")
        rv, ro = predict(r, x, val, device), predict(r, x, outer, device); del r; torch.cuda.empty_cache()
        d = train_independent(DenseNet121_3D(), x, y, train, 2, 14, 3e-4, device, SEED + fold, f"original_dense_f{fold}")
        dv, do = predict(d, x, val, device), predict(d, x, outer, device); del d; torch.cuda.empty_cache()
        w0 = float(minimize_scalar(lambda w: log_loss(y[val], w * rv + (1 - w) * dv, labels=[0, 1]), bounds=(0, 1), method="bounded").x)
        val_e, outer_e = w0 * rv + (1 - w0) * dv, w0 * ro + (1 - w0) * do
        original = calibrated(y[val], val_e, outer_e)
        dr, dd = train_dml(x, y, train, device, SEED + 100 + fold, fold)
        drv, dro = predict(dr, x, val, device), predict(dr, x, outer, device)
        ddv, ddo = predict(dd, x, val, device), predict(dd, x, outer, device)
        del dr, dd; torch.cuda.empty_cache()
        wd = float(minimize_scalar(lambda w: log_loss(y[val], w * drv + (1 - w) * ddv, labels=[0, 1]), bounds=(0, 1), method="bounded").x)
        val_d, outer_d = wd * drv + (1 - wd) * ddv, wd * dro + (1 - wd) * ddo
        dml = calibrated(y[val], val_d, outer_d)
        fm = {"fold": fold, "original_flip": metric(y[outer], original), "dml_flip": metric(y[outer], dml), "original_resnet_weight": w0, "dml_resnet_weight": wd}
        folds.append(fm); print(json.dumps(fm), flush=True)
        for j, idx in enumerate(outer): all_rows.append({"uid": rows[idx]["uid"], "actual": int(y[idx]), "fold": fold, "original_flip": float(original[j]), "dml_flip": float(dml[j])})
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n")
    truth = np.asarray([r["actual"] for r in all_rows]); op = np.asarray([r["original_flip"] for r in all_rows]); dp = np.asarray([r["dml_flip"] for r in all_rows])
    summary = {"folds": folds, "oof_original_flip": metric(truth, op), "oof_dml_flip": metric(truth, dp)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
