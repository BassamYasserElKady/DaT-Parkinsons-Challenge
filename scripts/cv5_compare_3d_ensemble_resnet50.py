#!/usr/bin/env python3
"""Nested five-fold comparison of calibrated 3D ensemble and ResNet-50."""
from __future__ import annotations
import copy, csv, importlib.util, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_calibrated_3d_ensemble_vs_resnet50"
SEED = 20260931

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def resnet50_builder():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("downloaded_resnet50_cv", ROOT / "original_submission_files/model.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_model

def loader(x, y, ids, batch, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    xx = torch.from_numpy(np.asarray(x[ids], dtype=np.float32))
    return DataLoader(TensorDataset(xx, torch.from_numpy(y[ids])), batch_size=batch, shuffle=shuffle, generator=g, num_workers=0, pin_memory=True)

def predict(model, data, device):
    model.eval(); yy, pp = [], []
    with torch.no_grad():
        for x, y in data:
            yy.extend(y.numpy()); pp.extend(torch.softmax(model(x.to(device, non_blocking=True)), 1)[:, 1].cpu().numpy())
    return np.asarray(yy, dtype=int), np.asarray(pp, dtype=float)

def fit(model, x, y, train_ids, val_ids, outer_ids, cfg, device, seed):
    seed_all(seed); model = model.to(device)
    train = loader(x, y, train_ids, cfg["batch"], True, seed)
    val = loader(x, y, val_ids, cfg["batch"], False, seed)
    outer = loader(x, y, outer_ids, cfg["batch"], False, seed)
    counts = np.bincount(y[train_ids], minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.get("smooth", 0.0))
    cls = torch.optim.Adam if cfg.get("optimizer") == "Adam" else torch.optim.AdamW
    opt = cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    if cfg.get("cosine"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    else:
        mode = "max" if cfg["select"] == "auroc" else "min"
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -np.inf if cfg["select"] == "auroc" else np.inf; state = None; best_epoch = 0; stale = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if cfg.get("noise", 0): xb = xb + torch.randn_like(xb) * cfg["noise"]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"): loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        vy, vp = predict(model, val, device)
        value = roc_auc_score(vy, vp) if cfg["select"] == "auroc" else log_loss(vy, vp)
        if cfg.get("cosine"): sched.step()
        else: sched.step(value)
        improved = value > best if cfg["select"] == "auroc" else value < best
        if improved: best, state, best_epoch, stale = value, copy.deepcopy(model.state_dict()), epoch, 0
        else: stale += 1
        if stale >= cfg["patience"]: break
    model.load_state_dict(state); vy, vp = predict(model, val, device); oy, op = predict(model, outer, device)
    del model; torch.cuda.empty_cache()
    return vy, vp, oy, op, best_epoch

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))

def calibrate(val_y, val_p, outer_p):
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(val_p).reshape(-1, 1), val_y)
    return model.predict_proba(logit(outer_p).reshape(-1, 1))[:, 1], float(model.coef_[0, 0]), float(model.intercept_[0])

def metric(y, p):
    pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "sensitivity": float(tp/(tp+fn)), "specificity": float(tn/(tn+fp)), "f1": float(f1_score(y, pred)), "auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open()))
    y = np.asarray([int(r["is_pathologic"]) for r in rows]); development = np.flatnonzero(np.asarray([r["split"] for r in rows]) != "test")
    x3 = np.load(ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy", mmap_mode="r")[:, None]
    x2 = np.load(ROOT / "results/downloaded_resnet50_cbam/input_five_slices.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); build50 = resnet50_builder()
    configs = {
        "r18": {"batch": 8, "lr": 1e-3, "wd": 1e-4, "epochs": 40, "patience": 8, "select": "auroc"},
        "dense": {"batch": 2, "lr": 3e-4, "wd": 1e-4, "epochs": 40, "patience": 8, "select": "auroc"},
        "r50": {"batch": 16, "lr": 0.0013204812211585124, "wd": 8.069617784127901e-6, "epochs": 30, "patience": 6, "select": "logloss", "optimizer": "Adam", "smooth": 0.005642681480250881, "cosine": True, "noise": 0.01},
    }
    folds, all_rows = [], []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (outer_train_pos, outer_pos) in enumerate(skf.split(development, y[development]), 1):
        outer_train, outer = development[outer_train_pos], development[outer_pos]
        train, val = train_test_split(outer_train, test_size=.2, stratify=y[outer_train], random_state=SEED + fold)
        print(f"fold={fold}/5 train={len(train)} val={len(val)} outer={len(outer)}", flush=True)
        outputs = {}
        for name, model, xx in (("r18", ResNet18CBAM3D(), x3), ("dense", DenseNet121_3D(), x3), ("r50", build50(), x2)):
            outputs[name] = fit(model, xx, y, train, val, outer, configs[name], device, SEED + fold)
            print(f"fold={fold} model={name} best_epoch={outputs[name][4]}", flush=True)
        vy = outputs["r18"][0]; oy = outputs["r18"][2]
        vr, vd = outputs["r18"][1], outputs["dense"][1]; or_, od = outputs["r18"][3], outputs["dense"][3]
        opt = minimize_scalar(lambda w: log_loss(vy, w*vr+(1-w)*vd), bounds=(0,1), method="bounded")
        w = float(opt.x); ensemble_outer, a, b = calibrate(vy, w*vr+(1-w)*vd, w*or_+(1-w)*od)
        r50_outer, c, d = calibrate(outputs["r50"][0], outputs["r50"][1], outputs["r50"][3])
        em, rm = metric(oy, ensemble_outer), metric(oy, r50_outer)
        folds.append({"fold": fold, "ensemble": em, "resnet50": rm, "ensemble_resnet_weight": w, "ensemble_platt": [a,b], "resnet50_platt": [c,d]})
        for pos, index in enumerate(outer): all_rows.append({"uid": rows[index]["uid"], "actual": int(oy[pos]), "fold": fold, "ensemble_score": float(ensemble_outer[pos]), "resnet50_score": float(r50_outer[pos])})
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n")
        print(f"fold={fold} ensemble={em} resnet50={rm}", flush=True)
    truth = np.asarray([r["actual"] for r in all_rows]); ep = np.asarray([r["ensemble_score"] for r in all_rows]); rp = np.asarray([r["resnet50_score"] for r in all_rows])
    summary = {"n_development": len(development), "outer_folds": 5, "inner_validation_fraction": .2, "ensemble_oof": metric(truth, ep), "resnet50_oof": metric(truth, rp), "folds": folds}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "oof_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__": main()
