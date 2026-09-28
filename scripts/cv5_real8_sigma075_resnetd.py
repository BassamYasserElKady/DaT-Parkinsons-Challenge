#!/usr/bin/env python3
"""Five-fold ResNet-D + CBAM on existing sigma=0.75 adaptive 8-slice inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from cv5_original_flip_vs_dml import predict, seed_all, train_independent
from cv5_real8_interp16 import metrics, standardize_and_resize
from train_registered_consecutive_resnet18_cbam import BasicBlock3D, ResNet18CBAM3D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_sigma075_resnetd"
INPUT = ROOT / "results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
SEED = 20260931


class ResNetDBlock3D(BasicBlock3D):
    """Basic block with a ResNet-D anti-aliased projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1)) -> None:
        super().__init__(in_channels, out_channels, stride)
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.AvgPool3d(kernel_size=stride, stride=stride, ceil_mode=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )


class ResNet18CBAM3D_D(ResNet18CBAM3D):
    def _make_layer(self, channels: int, blocks: int, stride) -> nn.Sequential:
        layers = [ResNetDBlock3D(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(ResNetDBlock3D(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader(ROWS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = np.load(INPUT).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = ResNet18CBAM3D_D()
    print(f"device={device} input_shape={native.shape}", flush=True)
    print(f"isotropic_downsample={probe.layer2[0].downsample}", flush=True)
    print(
        f"anisotropic_probe={ResNetDBlock3D(64, 128, (1, 2, 2)).downsample}",
        flush=True,
    )

    splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y)
    oof = np.full(len(y), np.nan)
    fold_ids = np.empty(len(y), dtype=np.int64)
    folds = []
    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_ids[outer_index] = fold
        x, means, stds = standardize_and_resize(native, train_index)
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        model = train_independent(
            ResNet18CBAM3D_D(), x, y, train_index, 8, 16, 1e-3,
            device, SEED + fold, f"sigma075_resnetd_f{fold}",
        )
        oof[outer_index] = predict(model, x, outer_index, device)
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        record = {
            "fold": fold,
            "train": int(len(train_index)),
            "outer": int(len(outer_index)),
            "channel_means": means.tolist(),
            "channel_stds": stds.tolist(),
            "resnetd": metrics(y[outer_index], oof[outer_index]),
        }
        folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "oof_progress.npz", probability=oof, fold=fold_ids)
        print(json.dumps(record), flush=True)

    summary = {
        "n_scans": len(y),
        "architecture": "3D ResNet-18-D + CBAM",
        "shortcut": "AvgPool3d(stride, ceil_mode=True) -> Conv3d(1x1x1, stride=1) -> BatchNorm3d",
        "input": "8 adaptive axial slices after post-registration Gaussian smoothing sigma=0.75, interpolated to 16x64x64",
        "training_augmentation": "random horizontal flip only",
        "epochs": 16,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "resnetd_oof": metrics(y, oof),
        "folds": folds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "fold", "resnetd"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "uid": row["uid"], "actual": int(y[index]),
                "fold": int(fold_ids[index]), "resnetd": float(oof[index]),
            })
    print(json.dumps(summary["resnetd_oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
