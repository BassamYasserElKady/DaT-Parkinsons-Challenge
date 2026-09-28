#!/usr/bin/env python3
"""Five-fold evaluation of the 8-slice ensemble with CBAM on both branches."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from cv5_original_flip_vs_dml import (
    ResNet18CBAM3D,
    predict,
    seed_all,
    train_independent,
)
from train_registered_consecutive_densenet121_3d import (
    _DenseBlock3D,
    _Transition3D,
)
from train_registered_consecutive_resnet18_cbam import CBAM3D
from cv5_real8_interp16 import build_inputs, metrics, standardize_and_resize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_dense_cbam"
ROWS_PATH = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
INPUT_PATH = ROOT / "results/cv5_real8_interp16/real8_native_per_volume_zscore.npy"
SEED = 20260931
TARGET_SHAPE = (16, 64, 64)


class DenseNet121CBAM3D(nn.Module):
    """The existing 3D DenseNet-121 with CBAM after every dense block."""

    def __init__(self, num_classes: int = 2, growth_rate: int = 32, block_config: tuple[int, ...] = (6, 12, 24, 16), num_init_features: int = 64) -> None:
        super().__init__()
        features = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv3d(1, num_init_features, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)),
                ("norm0", nn.BatchNorm3d(num_init_features)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
            ])
        )
        num_features = num_init_features
        for block_index, num_layers in enumerate(block_config):
            block_out = num_features + num_layers * growth_rate
            features.add_module(f"denseblock{block_index + 1}", _DenseBlock3D(num_layers, num_features, growth_rate))
            features.add_module(f"cbam{block_index + 1}", CBAM3D(block_out))
            num_features = block_out
            if block_index != len(block_config) - 1:
                transition_out = num_features // 2
                features.add_module(f"transition{block_index + 1}", _Transition3D(num_features, transition_out))
                num_features = transition_out
        features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features = features
        self.classifier = nn.Linear(num_features, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.features(x))
        x = torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)
        return self.classifier(x)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader(ROWS_PATH.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = np.load(INPUT_PATH).astype(np.float32, copy=False)
    if len(native) != len(y):
        raise RuntimeError(f"input/label mismatch: {len(native)} vs {len(y)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y))
    fold_id = np.empty(len(y), dtype=np.int64)
    resnet_oof = np.full(len(y), np.nan)
    dense_oof = np.full(len(y), np.nan)
    folds = []
    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_id[outer_index] = fold
        x, means, stds = standardize_and_resize(native, train_index)
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        resnet = train_independent(ResNet18CBAM3D(), x, y, train_index, 8, 16, 1e-3, device, SEED + fold, f"dense_cbam_resnet_f{fold}")
        resnet_oof[outer_index] = predict(resnet, x, outer_index, device)
        del resnet
        torch.cuda.empty_cache()
        dense = train_independent(DenseNet121CBAM3D(), x, y, train_index, 2, 14, 3e-4, device, SEED + fold, f"dense_cbam_densenet_f{fold}")
        dense_oof[outer_index] = predict(dense, x, outer_index, device)
        del dense, x
        torch.cuda.empty_cache()
        ensemble = 0.5 * resnet_oof[outer_index] + 0.5 * dense_oof[outer_index]
        record = {
            "fold": fold,
            "train": int(len(train_index)),
            "outer": int(len(outer_index)),
            "resnet_cbam": metrics(y[outer_index], resnet_oof[outer_index]),
            "densenet_cbam": metrics(y[outer_index], dense_oof[outer_index]),
            "equal_ensemble": metrics(y[outer_index], ensemble),
        }
        folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(OUT / "oof_progress.npz", resnet=resnet_oof, densenet=dense_oof, fold=fold_id)
        print(json.dumps(record), flush=True)
    ensemble_oof = 0.5 * resnet_oof + 0.5 * dense_oof
    clipped = np.clip(ensemble_oof, 0.005, 0.995)
    summary = {
        "n_scans": len(y),
        "input": "8 real consecutive slices linearly interpolated to 16",
        "training_augmentation": "horizontal flip only",
        "resnet": "3D ResNet-18 with CBAM after every residual block",
        "densenet": "3D DenseNet-121 with CBAM after every dense block",
        "ensemble_weighting": "0.5 ResNet + 0.5 DenseNet",
        "resnet_oof": metrics(y, resnet_oof),
        "densenet_cbam_oof": metrics(y, dense_oof),
        "equal_ensemble_oof": metrics(y, ensemble_oof),
        "equal_ensemble_clipped_005_995_oof": metrics(y, clipped),
        "folds": folds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "fold", "resnet", "densenet_cbam", "equal_ensemble", "equal_ensemble_clipped"])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"uid": row["uid"], "actual": int(y[i]), "fold": int(fold_id[i]), "resnet": float(resnet_oof[i]), "densenet_cbam": float(dense_oof[i]), "equal_ensemble": float(ensemble_oof[i]), "equal_ensemble_clipped": float(clipped[i])})
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("oof")}, indent=2), flush=True)
    print(f"output={OUT}", flush=True)


if __name__ == "__main__":
    main()
