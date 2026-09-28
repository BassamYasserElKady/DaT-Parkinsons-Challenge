#!/usr/bin/env python3
"""Train the auxiliary-head scratch Inception-v3 on all adaptive-eight MIPs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cv5_inception_v3_adaptive8_mip as base  # noqa: E402
from cv5_inception8_mip_auxiliary_ablation import make_model  # noqa: E402

DATA = ROOT / "results" / "cv5_inception_v3_adaptive8_mip" / "adaptive8_mips.npy"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
OUT = ROOT / "results" / "full_revised_four_model_submission"
SEED, EPOCHS, AUX_WEIGHT = 20260931, 16, 0.4


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); base.seed_all(SEED)
    rows = list(csv.DictReader(LABELS.open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    x = np.load(DATA).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required.")
    x = F.interpolate(torch.from_numpy(x)[:, None], size=(299, 299), mode="bilinear", align_corners=False)
    model = make_model(True).to(device)
    counts = np.bincount(y, minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader = DataLoader(TensorDataset(x, torch.from_numpy(y)), batch_size=16, shuffle=True, drop_last=True,
                        num_workers=0, generator=torch.Generator().manual_seed(SEED))
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if torch.rand(()) < .5: image = image.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            loss = criterion(output.logits, target) + AUX_WEIGHT * criterion(output.aux_logits, target)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(target)
        scheduler.step(); record = {"epoch": epoch, "loss": total / len(y), "learning_rate": optimizer.param_groups[0]["lr"]}; history.append(record)
        print(json.dumps(record), flush=True)
    torch.save({"inception_state_dict": model.cpu().state_dict()}, OUT / "auxiliary_inception8_mip_weights.pt")
    (OUT / "auxiliary_inception_training_metadata.json").write_text(json.dumps({"training_scan_count": len(y), "epochs": EPOCHS, "auxiliary_loss_weight": AUX_WEIGHT, "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": "CosineAnnealingLR", "augmentation": "random_horizontal_flip_only", "input": "adaptive eight-slice axial MIP resized to 299x299", "weights": "None (scratch)"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
