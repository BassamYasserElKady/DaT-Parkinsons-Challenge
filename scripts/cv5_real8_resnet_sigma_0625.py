#!/usr/bin/env python3
"""Run the ResNet-only post-registration Gaussian sigma=0.625 experiment."""

import csv
from pathlib import Path

import numpy as np
import torch

import cv5_real8_resnet_gaussian_ablation as ablation

ROOT = Path(__file__).resolve().parents[1]
ablation.OUT = ROOT / "results/cv5_real8_resnet_gaussian_ablation"
rows = list(csv.DictReader((ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open(newline="", encoding="utf-8")))
y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ablation.base.seed_all(ablation.SEED)
summary = ablation.run_sigma(0.625, rows, y, device)
print(summary, flush=True)
