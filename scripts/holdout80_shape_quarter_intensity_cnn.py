#!/usr/bin/env python3
"""Clean 80/20 3D-CNN test of contour shape quarters plus intensity maps."""
from __future__ import annotations

import csv
import json
import random
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ROOT / "results" / "adaptive8_3d_quarter_shapes" / "labels"
MAPS = ROOT / "results" / "adaptive8_sliding_intensity_pyramid" / "maps"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
OUT = ROOT / "results" / "holdout80_shape_quarter_intensity_cnn"
SEED = 20260931
EPOCHS = 16
EPS = 1e-6


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy": float(accuracy_score(y, p >= .5)), "incorrect": int(np.sum((p >= .5) != y))}


def make_input(uid: str) -> np.ndarray:
    """Create 10 shape + 6 intensity channels, then resize to D=8,H=W=64."""
    label = np.asarray(nib.load(str(SHAPES / f"{uid}_adaptive8_3d_quarters.nii.gz")).dataobj, dtype=np.uint8)
    channels = np.zeros((16, *label.shape), dtype=np.float32)
    channels[0] = (label >= 1) & (label <= 4)       # complete left 3D shape
    channels[1] = (label >= 5) & (label <= 8)       # complete right 3D shape
    for q in range(4):
        channels[2 + q] = label == 1 + q
        channels[6 + q] = label == 5 + q
    sparse = np.load(MAPS / f"{uid}_adaptive8_sliding_pyramid.npz")
    xyz, side = sparse["xyz"], sparse["side"]
    raw = sparse["intensity_1"].astype(np.float32)
    # Per-scan scale preserves the relative within-striatum distribution and
    # prevents scanner count scale from dominating the shape representation.
    positive = raw[raw > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    scale = max(scale, 1e-6)
    for level, key in enumerate(("intensity_1", "intensity_2", "intensity_3")):
        value = sparse[key].astype(np.float32) / scale
        for side_value, channel in ((0, 10 + level), (1, 13 + level)):
            selected = side == side_value
            if np.any(selected):
                coordinate = xyz[selected]
                channels[channel, coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]] = value[selected]
    # NIfTI arrangement x,y,z -> neural-net arrangement depth(z), height(y), width(x).
    tensor = torch.from_numpy(channels.transpose(0, 3, 2, 1)).unsqueeze(0)
    return F.interpolate(tensor, size=(8, 64, 64), mode="trilinear", align_corners=False).squeeze(0).numpy().astype(np.float16)


class ShapeIntensityCNN(nn.Module):
    """Small regularized 3D model appropriate for 1,362 scans."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1, bias=False), nn.BatchNorm3d(64), nn.GELU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 96, 3, padding=1, bias=False), nn.BatchNorm3d(96), nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.40), nn.Linear(96, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def train_predict(x: np.ndarray, y: np.ndarray, development: np.ndarray, test: np.ndarray, device: torch.device) -> np.ndarray:
    model = ShapeIntensityCNN().to(device)
    counts = np.bincount(y[development], minlength=2).astype(float)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[development]), torch.from_numpy(y[development])),
                              batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                              generator=torch.Generator().manual_seed(SEED))
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(batch_x), batch_y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += float(loss.detach()) * len(batch_y)
        scheduler.step(); print(f"epoch={epoch}/{EPOCHS} loss={total/len(development):.6f}", flush=True)
    model.eval(); output = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x[test])), batch_size=16, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for (batch_x,) in loader:
            output.append(torch.softmax(model(batch_x.to(device, dtype=torch.float32)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(output), model.cpu()


def main() -> None:
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open()))
    uids = [row["uid"] for row in rows]; y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    development, test = next(StratifiedShuffleSplit(1, test_size=.20, random_state=SEED).split(np.zeros(len(y)), y))
    development, test = np.sort(development), np.sort(test)
    print(f"building_inputs scans={len(y)}", flush=True)
    x = np.empty((len(y), 16, 8, 64, 64), dtype=np.float16)
    for index, uid in enumerate(uids, 1):
        x[index - 1] = make_input(uid)
        if index % 100 == 0 or index == len(y): print(f"inputs={index}/{len(y)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "progress.json").write_text(json.dumps({"stage": "training", "device": str(device), "development_scans": len(development), "test_scans": len(test)}, indent=2) + "\n")
    probability, model = train_predict(x, y, development, test, device)
    result = metrics(y[test], probability)
    summary = {"experiment": "clean stratified 80/20 holdout; outer test set unused during fitting", "input": "16 channels: bilateral full 3D shapes, eight PCA-quarter masks, and six three-level same-side intensity maps", "shape_source": "adaptive eight-slice independent contours", "development_scans": int(len(development)), "test_scans": int(len(test)), "epochs": EPOCHS, "test": result}
    np.savez_compressed(OUT / "test_predictions.npz", uid=np.asarray([uids[i] for i in test]), y=y[test], probability=probability)
    torch.save({"state_dict": model.state_dict()}, OUT / "model.pt")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "progress.json").write_text(json.dumps({"stage": "complete", **summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try: main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True); (OUT / "error.txt").write_text(traceback.format_exc()); raise
