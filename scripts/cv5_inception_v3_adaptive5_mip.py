#!/usr/bin/env python3
"""Five-fold scratch Inception-v3 evaluation on adaptive-five-slice MIPs."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3

ROOT = Path(__file__).resolve().parents[1]
MIPS = ROOT / "results/adaptive5_mip_2d/adaptive5_mips.npy"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OUT = ROOT / "results/cv5_inception_v3_adaptive5_mip"
SEED = 20260931
EPOCHS = 16


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "incorrect": int(np.sum((p >= 0.5) != y))}


def make_model() -> nn.Module:
    model = inception_v3(weights=None, aux_logits=False, num_classes=2, init_weights=True)
    old = model.Conv2d_1a_3x3
    model.Conv2d_1a_3x3.conv = nn.Conv2d(1, old.conv.out_channels,
                                         old.conv.kernel_size,
                                         stride=old.conv.stride,
                                         padding=old.conv.padding,
                                         bias=False)
    return model


def prepare(x: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(x.astype(np.float32, copy=False))[:, None]
    tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
    return tensor.to(device)


def train_fold(x, y, train_idx, test_idx, fold, device):
    seed_all(SEED + fold)
    model = make_model().to(device)
    class_counts = np.bincount(y[train_idx], minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_counts.sum() / (2 * class_counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_x, test_x = prepare(x[train_idx], device), prepare(x[test_idx], device)
    train_y = torch.from_numpy(y[train_idx]).to(device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED + fold), num_workers=0)
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(yb)
        scheduler.step()
        print(f"fold={fold} epoch={epoch}/{EPOCHS} loss={total/len(train_idx):.6f} lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
    model.eval(); predictions=[]
    with torch.inference_mode():
        for start in range(0, len(test_idx), 32):
            predictions.append(torch.softmax(model(test_x[start:start+32]), 1)[:, 1].cpu().numpy())
    del train_x, test_x, train_y, loader, model
    if device.type == "cuda": torch.cuda.empty_cache()
    return np.concatenate(predictions)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); seed_all(SEED)
    x = np.load(MIPS).astype(np.float32, copy=False)
    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(r["is_pathologic"]) for r in rows], dtype=np.int64)
    if x.shape != (1362, 109, 91) or len(y) != len(x):
        raise RuntimeError(f"unexpected MIP/label shapes: {x.shape}, {len(y)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.full(len(y), np.nan); records=[]
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), 1):
        p = train_fold(x, y, train_idx, test_idx, fold, device); oof[test_idx] = p
        rec = {"fold": fold, "train": int(len(train_idx)), "test": int(len(test_idx)), **metric(y[test_idx], p)}
        records.append(rec); (OUT/"progress.json").write_text(json.dumps(records, indent=2)+"\n"); print(json.dumps(rec), flush=True)
    summary = {"model":"Inception-v3 scratch, grayscale adaptive-five-slice axial MIP", "weights":"None (not ImageNet)", "epochs":EPOCHS, "input_shape":[1,299,299], "folds":records, "pooled":metric(y,oof)}
    np.savez(OUT/"oof_predictions.npz", y=y, prediction=oof)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
