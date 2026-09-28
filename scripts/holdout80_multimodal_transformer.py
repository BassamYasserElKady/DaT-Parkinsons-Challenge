#!/usr/bin/env python3
"""Clean 80/20 prototype of ResNet spatial tokens fused with SBR/morphology tokens.

The outer 20% test split is never used to fit image normalisation, tabular
standardisation, network weights, or any model-selection parameter.  This is
an intentionally small hybrid model suitable for a cohort of 1,362 scans;
it is not a native 3D ViT.
"""
from __future__ import annotations

import csv
import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "holdout80_multimodal_transformer"
WINDOWS = ROOT / "results" / "downloaded_rigid_adaptive_3d_ensemble" / "input_windows.npy"
FEATURES = ROOT / "results" / "cv5_svm_occipital_sbr" / "features.npz"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
SEED = 20260931
EPOCHS = 16
EPS = 1e-6


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "incorrect": int(np.sum((p >= 0.5) != y)),
    }


def prepare_images(native: np.ndarray, development: np.ndarray) -> np.ndarray:
    """Fit channel standardisation on development scans and resize all images."""
    mean = native[development].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = native[development].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    result = np.empty((len(native), 16, 64, 64), dtype=np.float32)
    zoom = (16 / native.shape[1], 64 / native.shape[2], 64 / native.shape[3])
    for index in range(len(native)):
        standardized = (native[index] - mean[:, None, None]) / std[:, None, None]
        result[index] = ndimage.zoom(standardized, zoom, order=1)
    return result


class SpatialResNet18CBAM(ResNet18CBAM3D):
    """Existing 3D ResNet-18+CBAM backbone exposing layer-4 spatial features."""

    def spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class MultimodalTransformer(nn.Module):
    """Fuse ResNet spatial tokens with four semantically grouped tabular tokens."""

    def __init__(self, d_model: int = 128, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.backbone = SpatialResNet18CBAM()
        self.image_projection = nn.Conv3d(512, d_model, kernel_size=1, bias=False)
        # 32-feature ordering is fixed by cv5_svm_occipital_sbr.py.
        self.left_projection = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.right_projection = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.global_projection = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.morphology_projection = nn.Sequential(nn.Linear(22, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, 5, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=256, dropout=0.25,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.30), nn.Linear(d_model, 2))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        feature_map = self.image_projection(self.backbone.spatial_features(image))
        spatial = feature_map.flatten(2).transpose(1, 2)
        # SBR columns: L caudate=24, R caudate=25, L putamen=26, R putamen=27.
        left = torch.stack((tabular[:, 24], tabular[:, 26]), dim=1)
        right = torch.stack((tabular[:, 25], tabular[:, 27]), dim=1)
        # Occipital mean/std, bilateral mean SBRs, and two asymmetry measures.
        global_features = tabular[:, 22:24]
        global_features = torch.cat((global_features, tabular[:, 28:32]), dim=1)
        tab_tokens = torch.stack((
            self.left_projection(left), self.right_projection(right),
            self.global_projection(global_features), self.morphology_projection(tabular[:, :22]),
        ), dim=1)
        cls = self.class_token.expand(image.shape[0], -1, -1)
        sequence = torch.cat((cls, tab_tokens, spatial), dim=1)
        # Learned positions distinguish CLS and the four semantic tabular tokens.
        sequence[:, :5] = sequence[:, :5] + self.position
        return self.head(self.transformer(sequence)[:, 0])


def loader(images: np.ndarray, tabular: np.ndarray, y: np.ndarray, ids: np.ndarray, batch: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(images[ids]).unsqueeze(1), torch.from_numpy(tabular[ids]), torch.from_numpy(y[ids])
    )
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


def train(model: nn.Module, images: np.ndarray, tabular: np.ndarray, y: np.ndarray, development: np.ndarray, device: torch.device) -> nn.Module:
    model = model.to(device)
    counts = np.bincount(y[development], minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2.0 * counts), device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-4},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    batches = loader(images, tabular, y, development, batch=4, shuffle=True, seed=SEED)
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0
        for image, features, label in batches:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
            image, features, label = image.to(device, non_blocking=True), features.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(image, features), label)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer); grad_scaler.update()
            total_loss += float(loss.detach()) * len(label)
        scheduler.step()
        print(f"epoch={epoch}/{EPOCHS} loss={total_loss / len(development):.6f}", flush=True)
    return model.eval()


def predict(model: nn.Module, images: np.ndarray, tabular: np.ndarray, y: np.ndarray, ids: np.ndarray, device: torch.device) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for image, features, _ in loader(images, tabular, y, ids, batch=4, shuffle=False, seed=SEED):
            output = model(image.to(device, non_blocking=True), features.to(device, non_blocking=True))
            probabilities.append(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def main() -> None:
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = np.load(WINDOWS, mmap_mode="r")
    feature_data = np.load(FEATURES)
    tabular_raw = np.asarray(feature_data["occipital_sbr32"], dtype=np.float32)
    if native.shape[0] != len(y) or tabular_raw.shape != (len(y), 32):
        raise RuntimeError(f"Input mismatch: images={native.shape}, tabular={tabular_raw.shape}, labels={len(y)}")
    development, test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(native, y))
    development, test = np.sort(development), np.sort(test)
    images = prepare_images(native, development)
    standardizer = StandardScaler().fit(tabular_raw[development])
    tabular = standardizer.transform(tabular_raw).astype(np.float32)
    np.savez_compressed(OUT / "split_and_preprocessing.npz", development=development, test=test, y=y,
                        tabular_mean=standardizer.mean_, tabular_scale=standardizer.scale_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress = {"stage": "training", "device": str(device), "development_scans": int(len(development)), "test_scans": int(len(test)), "epochs": EPOCHS}
    (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps(progress), flush=True)
    model = train(MultimodalTransformer(), images, tabular, y, development, device)
    probability = predict(model, images, tabular, y, test, device)
    result = metrics(y[test], probability)
    summary = {
        "experiment": "one clean stratified 80/20 prototype; test partition untouched until final inference",
        "image_input": "registered adaptive five-slice windows, resized to 16x64x64",
        "tabular_input": "32 label-free SBR/occipital/morphometry features; StandardScaler fit on development only",
        "architecture": "3D ResNet-18+CBAM layer-4 spatial tokens + four tabular tokens + 2-layer 4-head TransformerEncoder (d_model=128)",
        "development_scans": int(len(development)), "test_scans": int(len(test)), "epochs": EPOCHS,
        "test": result,
    }
    np.savez_compressed(OUT / "test_predictions.npz", uid=np.asarray([rows[i]["uid"] for i in test]), y=y[test], probability=probability)
    torch.save({"state_dict": model.cpu().state_dict(), "tabular_mean": standardizer.mean_, "tabular_scale": standardizer.scale_}, OUT / "model.pt")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "progress.json").write_text(json.dumps({"stage": "complete", **summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "error.txt").write_text(traceback.format_exc())
        raise
