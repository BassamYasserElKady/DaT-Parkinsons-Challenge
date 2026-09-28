#!/usr/bin/env python3
"""Leakage-controlled five-fold OOF test of striatum-protected MIP augmentation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cv5_inception_v3_adaptive8_mip as base  # noqa: E402
from render_inception_mip_background_block_shuffle import (  # noqa: E402
    shuffle_blocks_to_random_contour_targets,
)

OUT = ROOT / "results" / "cv5_inception8_mip_blockshuffle_augmentation"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
BRAIN = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
SEED, EPOCHS = 20260931, 16
RANGES = ((0.03, 0.70), (0.10, 0.75), (0.20, 0.80))


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= .5)), "incorrect": int(np.sum((p >= .5) != y))}


def supports() -> np.ndarray:
    rows = list(csv.DictReader((base.OUT / "slice_selection.csv").open()))
    mask = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    return np.stack([mask[:, :, int(row["start"]):int(row["end"]) + 1].any(2).T for row in rows])


def make_augmented(images: np.ndarray, masks: np.ndarray, seed_offset: int) -> np.ndarray:
    src = ROOT / "submission_ensemble_sigma075_8slice_svm" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from svm_features import retain_top_bins, segment
    out = np.empty((len(images) * 4, *images.shape[1:]), dtype=np.float32)
    for i, (image, brain) in enumerate(zip(images, masks)):
        striatum = segment(retain_top_bins(image, brain), brain)
        mean = float(image[striatum].mean()) if striatum.any() else 0.0
        out[4 * i] = image
        for variant, (low, high) in enumerate(RANGES, 1):
            rng = np.random.default_rng(SEED + seed_offset * 100_000 + i * 10 + variant)
            out[4 * i + variant], _ = shuffle_blocks_to_random_contour_targets(
                image, brain & ~striatum, mean, low, high, rng
            )
        if (i + 1) % 200 == 0 or i + 1 == len(images):
            print(f"augment fold={seed_offset} scans={i + 1}/{len(images)}", flush=True)
    return out


def predict(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    result: list[np.ndarray] = []; model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            x = F.interpolate(torch.from_numpy(images[start:start + 16])[:, None], size=(299, 299), mode="bilinear", align_corners=False).to(device)
            result.append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def run_fold(images: np.ndarray, labels: np.ndarray, masks: np.ndarray, train: np.ndarray, test: np.ndarray, fold: int, device: torch.device) -> np.ndarray:
    base.seed_all(SEED + fold)
    augmented = make_augmented(images[train], masks[train], fold)
    augmented_y = np.repeat(labels[train], 4)
    model = base.make_model().to(device)
    counts = np.bincount(labels[train], minlength=2).astype(float)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader = DataLoader(TensorDataset(torch.from_numpy(augmented), torch.from_numpy(augmented_y)), batch_size=16, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED + fold))
    for epoch in range(1, EPOCHS + 1):
        model.train(); loss_sum = 0.0
        for x, y in loader:
            x = F.interpolate(x[:, None], size=(299, 299), mode="bilinear", align_corners=False).to(device); y = y.to(device)
            if torch.rand(()) < .5: x = x.flip(-1)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(x), y); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
        scheduler.step()
        print(f"fold={fold} epoch={epoch}/{EPOCHS} train_loss={loss_sum / len(augmented):.6f}", flush=True)
    prediction = predict(model, images[test], device)
    del model, augmented, augmented_y, loader
    torch.cuda.empty_cache()
    return prediction


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); base.seed_all(SEED)
    rows = list(csv.DictReader(LABELS.open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    x, m = base.build_data(rows), supports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required for this five-fold run.")
    oof = np.full(len(y), np.nan, dtype=np.float32); reports: list[dict[str, float | int]] = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(splitter.split(x, y), 1):
        prediction = run_fold(x, y, m, tr, te, fold, device); oof[te] = prediction
        report = {"fold": fold, "train": int(len(tr)), "test": int(len(te)), **metric(y[te], prediction)}
        reports.append(report); (OUT / "progress.json").write_text(json.dumps(reports, indent=2) + "\n")
        np.savez(OUT / "oof_partial.npz", y=y, prediction=oof)
        print(json.dumps(report), flush=True)
    summary = {"experiment": "scratch Inception-v3, adaptive eight-slice MIP, 2x2 background block augmentation", "evaluation": "five-fold stratified OOF; augmentation only within each training fold; untouched validation MIPs; fixed 16 epochs", "augmentation": {"variants": 3, "ranges": RANGES, "striatum_protected": True, "intracranial_only": True}, "folds": reports, "pooled": metric(y, oof)}
    np.savez(OUT / "oof_predictions.npz", uid=np.asarray([r["uid"] for r in rows]), y=y, prediction=oof)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
