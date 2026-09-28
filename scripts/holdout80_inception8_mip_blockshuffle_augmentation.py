#!/usr/bin/env python3
"""Single leakage-controlled holdout test of 2x2 background-block MIP augmentation.

Only the 80% training partition receives the three shuffled variants.  The 20%
holdout partition always uses the unmodified adaptive eight-slice MIP.
"""
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cv5_inception_v3_adaptive8_mip as base  # noqa: E402
from render_inception_mip_background_block_shuffle import (  # noqa: E402
    shuffle_blocks_to_random_contour_targets,
)

OUT = ROOT / "results" / "holdout80_inception8_mip_blockshuffle_augmentation"
LABELS = ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv"
BRAIN = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
SEED = 20260931
EPOCHS = 16
RANGES = ((0.03, 0.70), (0.10, 0.75), (0.20, 0.80))


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "incorrect": int(np.sum((probabilities >= 0.5) != labels)),
    }


def selected_brain_masks() -> np.ndarray:
    """Return the axial brain support for every scan's already-selected MIP."""
    audit = list(csv.DictReader((base.OUT / "slice_selection.csv").open()))
    brain = np.asarray(nib.load(str(BRAIN)).dataobj) > 0
    return np.stack(
        [brain[:, :, int(row["start"]): int(row["end"]) + 1].any(axis=2).T for row in audit]
    )


def segment_mip(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Use the existing SVM hotspot segmentation solely to protect the striatum."""
    src = ROOT / "submission_ensemble_sigma075_8slice_svm" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from svm_features import retain_top_bins, segment

    return segment(retain_top_bins(image, support), support)


def augment_training_images(images: np.ndarray, supports: np.ndarray) -> np.ndarray:
    """Return original plus three deterministic shuffled variants per training MIP."""
    output = np.empty((len(images) * 4, *images.shape[1:]), dtype=np.float32)
    for index, (image, support) in enumerate(zip(images, supports)):
        striatum = segment_mip(image, support)
        permitted = support & ~striatum
        contour_mean = float(image[striatum].mean()) if striatum.any() else 0.0
        output[index * 4] = image
        for variant, (low, high) in enumerate(RANGES, start=1):
            rng = np.random.default_rng(SEED + index * 100 + variant)
            shuffled, _ = shuffle_blocks_to_random_contour_targets(
                image, permitted, contour_mean, low, high, rng
            )
            output[index * 4 + variant] = shuffled
        if (index + 1) % 100 == 0 or index + 1 == len(images):
            print(f"augmented={index + 1}/{len(images)}", flush=True)
    return output


def predict(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            batch = torch.from_numpy(images[start:start + 16])[:, None]
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base.seed_all(SEED)
    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    labels = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    images = base.build_data(rows)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=SEED
    )
    supports = selected_brain_masks()
    augmented = augment_training_images(images[train_idx], supports[train_idx])
    augmented_labels = np.repeat(labels[train_idx], 4)
    np.save(OUT / "train_augmented_mips.npy", augmented)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.make_model().to(device)
    counts = np.bincount(labels[train_idx], minlength=2).astype(float)
    class_weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    dataset = TensorDataset(torch.from_numpy(augmented), torch.from_numpy(augmented_labels))
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = F.interpolate(batch_x[:, None], size=(299, 299), mode="bilinear", align_corners=False).to(device)
            batch_y = batch_y.to(device)
            if torch.rand(()) < 0.5:
                batch_x = batch_x.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(batch_y)
        scheduler.step()
        holdout = metrics(labels[test_idx], predict(model, images[test_idx], device))
        record = {"epoch": epoch, "train_loss": total_loss / len(augmented), **holdout}
        history.append(record)
        (OUT / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    final_probability = predict(model, images[test_idx], device)
    final = metrics(labels[test_idx], final_probability)
    best = min(history, key=lambda item: float(item["log_loss"]))
    np.savez(OUT / "holdout_predictions.npz", uid=np.asarray([rows[i]["uid"] for i in test_idx]), y=labels[test_idx], prediction=final_probability)
    summary = {
        "experiment": "scratch Inception-v3; adaptive 8-slice MIP; background 2x2 block augmentation",
        "split": {"seed": SEED, "train": int(len(train_idx)), "holdout": int(len(test_idx)), "stratified": True},
        "augmentation": {"train_only": True, "variants_per_original": 3, "ranges_of_contour_mean": RANGES, "striatum_protected": True, "brain_limited": True},
        "training": {"epochs": EPOCHS, "batch_size": 16, "flip_probability": 0.5, "device": str(device)},
        "final_epoch": final,
        "best_holdout_log_loss_epoch": best,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
