#!/usr/bin/env python3
"""Five-fold, matched auxiliary-classifier ablation for scratch Inception-v3."""
from __future__ import annotations

import argparse
import csv
import json
import sys
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
sys.path.insert(0, str(ROOT / "scripts"))
import cv5_inception_v3_adaptive8_mip as base  # noqa: E402

LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
SEED, EPOCHS, AUX_WEIGHT = 20260931, 16, 0.4


def make_model(auxiliary: bool) -> nn.Module:
    model = inception_v3(weights=None, aux_logits=auxiliary, num_classes=2, init_weights=True)
    first = model.Conv2d_1a_3x3
    first.conv = nn.Conv2d(1, first.conv.out_channels, first.conv.kernel_size,
                           stride=first.conv.stride, padding=first.conv.padding, bias=False)
    return model


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= .5)), "incorrect": int(np.sum((p >= .5) != y))}


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    result: list[np.ndarray] = []; model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 16):
            logits = model(x[start:start + 16].to(device))
            result.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def run_fold(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, fold: int, auxiliary: bool, device: torch.device) -> np.ndarray:
    base.seed_all(SEED + fold)
    train_x = F.interpolate(torch.from_numpy(x[train])[:, None], size=(299, 299), mode="bilinear", align_corners=False)
    test_x = F.interpolate(torch.from_numpy(x[test])[:, None], size=(299, 299), mode="bilinear", align_corners=False)
    counts = np.bincount(y[train], minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    model = make_model(auxiliary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader = DataLoader(TensorDataset(train_x, torch.from_numpy(y[train])), batch_size=16, shuffle=True,
                        drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(SEED + fold))
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            if torch.rand(()) < .5: batch = batch.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output.logits, target) + AUX_WEIGHT * criterion(output.aux_logits, target) if auxiliary else criterion(output, target)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(target)
        scheduler.step(); print(f"fold={fold} epoch={epoch}/{EPOCHS} train_loss={total / len(train):.6f}", flush=True)
    p = predict(model, test_x, device)
    del model, train_x, test_x, loader
    torch.cuda.empty_cache()
    return p


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--auxiliary", action="store_true"); args = parser.parse_args()
    out = ROOT / "results" / ("cv5_inception8_mip_auxiliary" if args.auxiliary else "cv5_inception8_mip_noauxiliary_matched")
    out.mkdir(parents=True, exist_ok=True); base.seed_all(SEED)
    rows = list(csv.DictReader(LABELS.open())); y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64); x = base.build_data(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required.")
    oof = np.full(len(y), np.nan, dtype=np.float32); reports = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y), 1):
        p = run_fold(x, y, tr, te, fold, args.auxiliary, device); oof[te] = p
        report = {"fold": fold, "train": int(len(tr)), "test": int(len(te)), **metric(y[te], p)}; reports.append(report)
        (out / "progress.json").write_text(json.dumps(reports, indent=2) + "\n"); np.savez(out / "oof_partial.npz", y=y, prediction=oof)
        print(json.dumps(report), flush=True)
    summary = {"model": "scratch Inception-v3", "auxiliary_classifier": args.auxiliary, "auxiliary_loss_weight": AUX_WEIGHT if args.auxiliary else 0.0, "evaluation": "five-fold stratified OOF; fixed 16 epochs; matched drop_last batching", "folds": reports, "pooled": metric(y, oof)}
    np.savez(out / "oof_predictions.npz", uid=np.asarray([r["uid"] for r in rows]), y=y, prediction=oof); (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
