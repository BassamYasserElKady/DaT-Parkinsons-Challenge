#!/usr/bin/env python3
from __future__ import annotations

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "holdout_600_250_512_dml_swap60"
SEED = 20260931
TEMPERATURE = 2.0
KL_WEIGHT = 0.5


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def dml_train(x, y, device):
    seed_all(SEED)
    resnet, densenet = ResNet18CBAM3D().to(device), DenseNet121_3D().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(SEED), pin_memory=True, num_workers=0)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weights)
    ropt = torch.optim.AdamW(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    dopt = torch.optim.AdamW(densenet.parameters(), lr=3e-4, weight_decay=1e-4)
    rs = torch.optim.lr_scheduler.CosineAnnealingLR(ropt, T_max=16)
    ds = torch.optim.lr_scheduler.CosineAnnealingLR(dopt, T_max=16)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, 17):
        resnet.train(); densenet.train(); total = 0.0
        for xb, yb in loader:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            ropt.zero_grad(set_to_none=True); dopt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                ra, da = resnet(xb), densenet(xb)
                p_r = F.log_softmax(ra / TEMPERATURE, dim=1); p_d = F.softmax(da / TEMPERATURE, dim=1)
                p_d_log = F.log_softmax(da / TEMPERATURE, dim=1); p_r_prob = F.softmax(ra / TEMPERATURE, dim=1)
                kl = (F.kl_div(p_r, p_d, reduction="batchmean") + F.kl_div(p_d_log, p_r_prob, reduction="batchmean")) * (TEMPERATURE ** 2) / 2
                loss = ce(ra, yb) + ce(da, yb) + KL_WEIGHT * kl
            scaler.scale(loss).backward(); scaler.step(ropt); scaler.step(dopt); scaler.update(); total += float(loss.detach()) * len(yb)
        rs.step(); ds.step(); print(f"dml epoch {epoch}/16: loss={total / len(y):.6f}", flush=True)
    return resnet.eval(), densenet.eval()


def predict(model, x, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1)), batch_size=4, shuffle=False, pin_memory=True)
    out = []
    with torch.inference_mode():
        for (xb,) in loader:
            out.append(torch.softmax(model(xb.to(device, non_blocking=True)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True); seed_all(SEED)
    x = np.load(ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy").astype(np.float32, copy=False)
    # Start from the deterministic 600/250/512 split, then swap 150 validation and 150 test scans.
    rows = list(csv.DictReader((ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open()))
    y = np.asarray([int(r["is_pathologic"]) for r in rows])
    indices = np.arange(len(y))
    train_val_idx, base_test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=512, random_state=SEED).split(indices, y))
    train_rel, base_val_rel = next(StratifiedShuffleSplit(n_splits=1, test_size=250, random_state=SEED + 1).split(train_val_idx, y[train_val_idx]))
    train_idx = train_val_idx[train_rel]
    base_val_idx = train_val_idx[base_val_rel]
    val_move_rel, val_keep_rel = next(StratifiedShuffleSplit(n_splits=1, train_size=150, random_state=SEED + 2).split(base_val_idx, y[base_val_idx]))
    test_fill_rel, test_remain_rel = next(StratifiedShuffleSplit(n_splits=1, train_size=150, random_state=SEED + 3).split(base_test_idx, y[base_test_idx]))
    val_idx = np.concatenate((base_val_idx[val_keep_rel], base_test_idx[test_fill_rel]))
    test_idx = np.concatenate((base_test_idx[test_remain_rel], base_val_idx[val_move_rel]))
    train = np.zeros(len(y), dtype=bool); val = np.zeros(len(y), dtype=bool); test = np.zeros(len(y), dtype=bool)
    train[train_idx] = True; val[val_idx] = True; test[test_idx] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet, densenet = dml_train(x[train], y[train], device)
    rv, rt = predict(resnet, x[val], device), predict(resnet, x[test], device)
    dv, dt = predict(densenet, x[val], device), predict(densenet, x[test], device)
    weight = float(minimize_scalar(lambda w: log_loss(y[val], w * rv + (1 - w) * dv, labels=[0, 1]), bounds=(0, 1), method="bounded").x)
    val_raw, test_raw = weight * rv + (1 - weight) * dv, weight * rt + (1 - weight) * dt
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=SEED).fit(np.log(np.clip(val_raw, 1e-6, 1 - 1e-6) / np.clip(1 - val_raw, 1e-6, 1)).reshape(-1, 1), y[val])
    test_cal = cal.predict_proba(np.log(np.clip(test_raw, 1e-6, 1 - 1e-6) / np.clip(1 - test_raw, 1e-6, 1)).reshape(-1, 1))[:, 1]
    metric = lambda yy, pp: {"auroc": float(roc_auc_score(yy, pp)), "log_loss": float(log_loss(yy, pp, labels=[0, 1])), "accuracy": float(accuracy_score(yy, pp >= .5))}
    report = {"temperature": TEMPERATURE, "kl_weight": KL_WEIGHT, "resnet_weight": weight, "densenet_weight": 1 - weight, "validation": metric(y[val], weight * rv + (1 - weight) * dv), "test_raw": metric(y[test], test_raw), "test_calibrated": metric(y[test], test_cal)}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
