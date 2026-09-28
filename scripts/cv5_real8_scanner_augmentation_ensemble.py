#!/usr/bin/env python3
"""Five-fold ensemble with training-only scanner-variation augmentation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

import cv5_real8_post_registration_gaussian as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_scanner_augmentation_ensemble"
ROWS_PATH = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
REGISTERED = ROOT / "results/ants_registration_mni"
SEED = 20260931


class ScannerAugmentedDataset(Dataset):
    def __init__(self, x, y, ids, seed):
        self.x = torch.from_numpy(x[ids]).unsqueeze(1)
        self.y = torch.from_numpy(y[ids])
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index].clone()
        # Simulate count noise on the standardized image using a conservative
        # count scale; the original validation/test arrays are never changed.
        if self.rng.random() < 0.5:
            rate = torch.clamp((x + 2.5) * 20.0, min=1.0)
            x = x + (torch.poisson(rate) - rate) / torch.sqrt(rate) * 0.08
        # Small integer translations approximate scanner positioning variation.
        if self.rng.random() < 0.5:
            shifts = [self.rng.randint(-1, 1) for _ in range(3)]
            x = torch.roll(x, shifts=shifts, dims=(-3, -2, -1))
        # Random isotropic PSF blur, sigma in voxels.
        if self.rng.random() < 0.5:
            sigma = self.rng.uniform(0.5, 1.5)
            radius = max(1, int(np.ceil(3.0 * sigma)))
            axis = torch.arange(-radius, radius + 1, dtype=x.dtype)
            kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
            kernel = kernel / kernel.sum()
            k3 = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
            k3 = k3[None, None]
            x = F.conv3d(x[None], k3, padding=radius)[0]
        return x, self.y[index]


def augmented_loader(x, y, ids, batch, shuffle, seed):
    ds = ScannerAugmentedDataset(x, y, ids, seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=0, pin_memory=True)


def metrics(y, p):
    return base.metrics(y, np.clip(p, 0.005, 0.995))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.REGISTERED = REGISTERED
    base.SIGMA = 0.0
    base.INPUT_PATH = OUT / "native_windows.npy"
    base.loader = augmented_loader
    base.seed_all(SEED)
    rows = list(csv.DictReader(ROWS_PATH.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = base.build_inputs(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} training_augmentation=scanner_variation", flush=True)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y))
    fold_id = np.empty(len(y), dtype=np.int64)
    resnet_oof = np.full(len(y), np.nan)
    dense_oof = np.full(len(y), np.nan)
    folds = []
    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_id[outer_index] = fold
        x, _, _ = base.standardize_and_resize(native, train_index)
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        r = base.train_independent(base.ResNet18CBAM3D(), x, y, train_index, 8, 16, 1e-3, device, SEED + fold, f"scanner_aug_resnet_f{fold}")
        resnet_oof[outer_index] = base.predict(r, x, outer_index, device)
        del r
        torch.cuda.empty_cache()
        d = base.train_independent(base.DenseNet121_3D(), x, y, train_index, 2, 14, 3e-4, device, SEED + fold, f"scanner_aug_densenet_f{fold}")
        dense_oof[outer_index] = base.predict(d, x, outer_index, device)
        del d, x
        torch.cuda.empty_cache()
        ensemble = 0.5 * resnet_oof[outer_index] + 0.5 * dense_oof[outer_index]
        record = {"fold": fold, "train": len(train_index), "outer": len(outer_index),
                  "resnet": metrics(y[outer_index], resnet_oof[outer_index]),
                  "densenet": metrics(y[outer_index], dense_oof[outer_index]),
                  "equal_ensemble": metrics(y[outer_index], ensemble)}
        folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "oof_progress.npz", resnet=resnet_oof, densenet=dense_oof, fold=fold_id)
        print(json.dumps(record), flush=True)
    ensemble_oof = 0.5 * resnet_oof + 0.5 * dense_oof
    summary = {"n_scans": len(y), "input": "8 registered adaptive slices",
               "training_augmentation": "horizontal flip plus conservative Poisson-like noise, small 3D translations, and random Gaussian PSF blur (sigma 0.5-1.5 voxels)",
               "ensemble_weighting": "0.5 ResNet + 0.5 DenseNet",
               "resnet_oof": metrics(y, resnet_oof), "densenet_oof": metrics(y, dense_oof),
               "equal_ensemble_oof": metrics(y, ensemble_oof), "folds": folds}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
