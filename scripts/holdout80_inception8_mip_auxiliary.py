#!/usr/bin/env python3
"""Scratch Inception-v3 with its BN auxiliary classifier, 80/20 holdout."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cv5_inception_v3_adaptive8_mip as base  # noqa: E402

OUT = ROOT / "results" / "holdout80_inception8_mip_auxiliary"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
SEED, EPOCHS, AUX_WEIGHT = 20260931, 16, 0.4


def make_model() -> nn.Module:
    """Random-init Inception-v3 with the standard BatchNorm auxiliary branch active."""
    model = inception_v3(weights=None, aux_logits=True, num_classes=2, init_weights=True)
    first = model.Conv2d_1a_3x3
    first.conv = nn.Conv2d(1, first.conv.out_channels, first.conv.kernel_size,
                           stride=first.conv.stride, padding=first.conv.padding, bias=False)
    return model


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= .5)), "incorrect": int(np.sum((p >= .5) != y))}


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval(); output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 16):
            logits = model(x[start:start + 16].to(device))
            output.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); base.seed_all(SEED)
    rows = list(csv.DictReader(LABELS.open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    raw = base.build_data(rows)
    index = np.arange(len(y))
    train, test = train_test_split(index, test_size=.20, stratify=y, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required for this experiment.")
    x_train = F.interpolate(torch.from_numpy(raw[train])[:, None], size=(299, 299), mode="bilinear", align_corners=False)
    x_test = F.interpolate(torch.from_numpy(raw[test])[:, None], size=(299, 299), mode="bilinear", align_corners=False)
    counts = np.bincount(y[train], minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader = DataLoader(TensorDataset(x_train, torch.from_numpy(y[train])), batch_size=16, shuffle=True,
                        drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(SEED))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if torch.rand(()) < .5: image = image.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            loss = criterion(outputs.logits, target) + AUX_WEIGHT * criterion(outputs.aux_logits, target)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(target)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / len(train), **metric(y[test], predict(model, x_test, device))}
        history.append(record); (OUT / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    final = predict(model, x_test, device)
    summary = {"model": "scratch Inception-v3 with BatchNorm auxiliary classifier", "split": {"train": int(len(train)), "holdout": int(len(test)), "seed": SEED, "stratified": True}, "training": {"epochs": EPOCHS, "main_loss_weight": 1.0, "auxiliary_loss_weight": AUX_WEIGHT, "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": "CosineAnnealingLR", "flip_probability": .5}, "final_epoch": metric(y[test], final), "best_holdout_log_loss_epoch": min(history, key=lambda item: float(item["log_loss"]))}
    np.savez(OUT / "holdout_predictions.npz", uid=np.asarray([rows[i]["uid"] for i in test]), y=y[test], prediction=final)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
