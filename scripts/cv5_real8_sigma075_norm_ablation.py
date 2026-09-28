#!/usr/bin/env python3
"""Five-fold GroupNorm/InstanceNorm ablation for 3D ResNet-18 + CBAM."""

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
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_sigma075_norm_ablation"
INPUT = ROOT / "results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
SEED = 20260931
VARIANTS = ("groupnorm", "instancenorm")


def replace_batchnorm(module: nn.Module, variant: str) -> nn.Module:
    """Recursively replace every BatchNorm3d layer, including shortcuts/stem."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            channels = child.num_features
            if variant == "groupnorm":
                groups = min(8, channels)
                while channels % groups:
                    groups -= 1
                replacement = nn.GroupNorm(groups, channels, eps=child.eps, affine=True)
            elif variant == "instancenorm":
                replacement = nn.InstanceNorm3d(
                    channels, eps=child.eps, affine=True, track_running_stats=False
                )
            else:
                raise ValueError(variant)
            setattr(module, name, replacement)
        else:
            replace_batchnorm(child, variant)
    return module


def make_model(variant: str) -> nn.Module:
    return replace_batchnorm(ResNet18CBAM3D(), variant)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader(ROWS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = np.load(INPUT).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = {variant: make_model(variant) for variant in VARIANTS}
    for variant, model in probe.items():
        bn_count = sum(isinstance(layer, nn.BatchNorm3d) for layer in model.modules())
        norm_count = sum(
            isinstance(layer, (nn.GroupNorm, nn.InstanceNorm3d))
            for layer in model.modules()
        )
        has_avgpool_shortcut = any(
            isinstance(layer, nn.AvgPool3d)
            for layer in model.layer2[0].downsample.modules()
        )
        print(
            f"variant={variant} batchnorm={bn_count} replacement_norm={norm_count} "
            f"avgpool_shortcut={has_avgpool_shortcut}", flush=True
        )
    del probe

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y))
    probabilities = {variant: np.full(len(y), np.nan) for variant in VARIANTS}
    fold_ids = np.empty(len(y), dtype=np.int64)
    folds = []
    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_ids[outer_index] = fold
        x, means, stds = standardize_and_resize(native, train_index)
        record = {
            "fold": fold,
            "train": int(len(train_index)),
            "outer": int(len(outer_index)),
            "channel_means": means.tolist(),
            "channel_stds": stds.tolist(),
        }
        print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        for variant in VARIANTS:
            model = train_independent(
                make_model(variant), x, y, train_index, 8, 16, 1e-3,
                device, SEED + fold, f"sigma075_{variant}_f{fold}",
            )
            probabilities[variant][outer_index] = predict(model, x, outer_index, device)
            record[variant] = metrics(y[outer_index], probabilities[variant][outer_index])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"variant_complete={variant} metrics={json.dumps(record[variant])}", flush=True)
        del x
        folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        np.savez(
            OUT / "oof_progress.npz", fold=fold_ids,
            **{variant: value for variant, value in probabilities.items()},
        )

    summary = {
        "n_scans": int(len(y)),
        "architecture": "3D ResNet-18 + CBAM; standard strided projection shortcut",
        "input": "8 adaptive axial slices after post-registration Gaussian sigma=0.75; resized to 16x64x64",
        "epochs": 16,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR(T_max=16)",
        "training_augmentation": "random horizontal flip only",
        "groupnorm": "8 groups; affine=True",
        "instancenorm": "affine=True; track_running_stats=False",
        "oof_metrics": {
            variant: metrics(y, probability)
            for variant, probability in probabilities.items()
        },
        "folds": folds,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["uid", "actual", "fold", *VARIANTS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "uid": row["uid"], "actual": int(y[index]), "fold": int(fold_ids[index]),
                **{variant: float(probabilities[variant][index]) for variant in VARIANTS},
            })
    print(json.dumps(summary["oof_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
