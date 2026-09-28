#!/usr/bin/env python3
"""Nested five-fold epoch selection for original 3D ResNet-18 + CBAM.

For each outer fold, an inner validation split selects the epoch with minimum
validation log loss over a 40-epoch trajectory. The network is then reset and
trained on the complete outer-training fold for exactly the selected duration
before one evaluation on the untouched outer fold.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv5_original_flip_vs_dml import class_loss, loader, predict, seed_all
from cv5_real8_interp16 import metrics, standardize_and_resize
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_sigma075_resnet18_cbam_nested_epochs"
INPUT = ROOT / "results/cv5_real8_post_registration_gaussian/real8_native_post_registration_gaussian.npy"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
SEED = 20260931
MAX_EPOCHS = 40
INNER_VALIDATION_FRACTION = 0.20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def train_epoch(model, dl, criterion, optimizer, scaler, device) -> float:
    model.train()
    total = 0.0
    for xb, yb in dl:
        if torch.rand(()) < 0.5:
            xb = xb.flip(-1)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(yb)
    return total / len(dl.dataset)


def select_epoch(x, y, train_ids, validation_ids, device, seed, fold):
    seed_all(seed)
    model = ResNet18CBAM3D().to(device)
    dl = loader(x, y, train_ids, BATCH_SIZE, True, seed)
    criterion = class_loss(y, train_ids, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_epoch = 1
    best_loss = float("inf")
    best_path = OUT / f"fold_{fold}_best_inner_checkpoint.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, dl, criterion, optimizer, scaler, device)
        validation_probability = predict(model, x, validation_ids, device)
        validation_metrics = metrics(y[validation_ids], validation_probability)
        current_loss = float(log_loss(
            y[validation_ids], validation_probability, labels=[0, 1]
        ))
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_weighted_ce": train_loss,
            "validation": validation_metrics,
        }
        history.append(row)
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "validation_log_loss": current_loss,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                },
                best_path,
            )
        print(
            f"fold={fold}/5 selection_epoch={epoch}/{MAX_EPOCHS} "
            f"train_ce={train_loss:.6f} val_logloss={current_loss:.6f} "
            f"val_auroc={validation_metrics['auroc']:.6f} best={best_epoch}",
            flush=True,
        )
        (OUT / f"fold_{fold}_selection_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_loss, history, str(best_path)


def fit_full_outer_training(x, y, train_ids, selected_epochs, device, seed, fold):
    seed_all(seed)
    model = ResNet18CBAM3D().to(device)
    dl = loader(x, y, train_ids, BATCH_SIZE, True, seed)
    criterion = class_loss(y, train_ids, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Preserve the same LR trajectory used during inner epoch selection.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses = []
    for epoch in range(1, selected_epochs + 1):
        train_loss = train_epoch(model, dl, criterion, optimizer, scaler, device)
        scheduler.step()
        losses.append(train_loss)
        print(
            f"fold={fold}/5 final_epoch={epoch}/{selected_epochs} "
            f"train_ce={train_loss:.6f}", flush=True
        )
    return model.eval(), losses


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    rows = list(csv.DictReader(ROWS.open(newline="", encoding="utf-8")))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = np.load(INPUT).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} scans={len(y)} max_epochs={MAX_EPOCHS} "
        f"inner_fraction={INNER_VALIDATION_FRACTION}", flush=True
    )

    outer_splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.empty(len(y), dtype=np.int64)
    folds = []

    for fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        inner_train, inner_validation = train_test_split(
            outer_train,
            test_size=INNER_VALIDATION_FRACTION,
            stratify=y[outer_train],
            random_state=SEED + fold,
        )
        fold_ids[outer_test] = fold
        print(
            f"fold={fold}/5 inner_train={len(inner_train)} "
            f"inner_validation={len(inner_validation)} outer_test={len(outer_test)}",
            flush=True,
        )

        # Inner preprocessing is learned without seeing inner validation or outer test.
        inner_x, inner_means, inner_stds = standardize_and_resize(native, inner_train)
        best_epoch, best_inner_loss, history, checkpoint = select_epoch(
            inner_x, y, inner_train, inner_validation, device, SEED + fold, fold
        )
        del inner_x

        # Refit preprocessing and model using every sample in the outer training fold.
        outer_x, outer_means, outer_stds = standardize_and_resize(native, outer_train)
        model, final_losses = fit_full_outer_training(
            outer_x, y, outer_train, best_epoch, device, SEED + fold, fold
        )
        oof[outer_test] = predict(model, outer_x, outer_test, device)
        outer_metrics = metrics(y[outer_test], oof[outer_test])
        del model, outer_x
        if device.type == "cuda":
            torch.cuda.empty_cache()

        record = {
            "fold": fold,
            "inner_train": int(len(inner_train)),
            "inner_validation": int(len(inner_validation)),
            "outer_train": int(len(outer_train)),
            "outer_test": int(len(outer_test)),
            "selected_epoch": int(best_epoch),
            "best_inner_validation_log_loss": float(best_inner_loss),
            "best_inner_checkpoint": checkpoint,
            "inner_channel_means": inner_means.tolist(),
            "inner_channel_stds": inner_stds.tolist(),
            "outer_channel_means": outer_means.tolist(),
            "outer_channel_stds": outer_stds.tolist(),
            "final_training_losses": final_losses,
            "outer_metrics": outer_metrics,
        }
        folds.append(record)
        (OUT / "progress.json").write_text(
            json.dumps(folds, indent=2) + "\n", encoding="utf-8"
        )
        np.savez(OUT / "oof_progress.npz", probability=oof, fold=fold_ids)
        print(f"fold_complete={json.dumps(record)}", flush=True)

    summary = {
        "n_scans": int(len(y)),
        "architecture": "3D ResNet-18 + CBAM (standard strided projection shortcut)",
        "input": "8 adaptive axial slices, post-registration Gaussian sigma=0.75, resized to 16x64x64",
        "outer_folds": 5,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "maximum_epochs": MAX_EPOCHS,
        "epoch_selection": "minimum inner-validation log loss",
        "final_fit": "reinitialized and retrained on complete outer-training fold for selected epoch count",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": f"CosineAnnealingLR(T_max={MAX_EPOCHS})",
        "training_augmentation": "random horizontal flip only",
        "oof_metrics": metrics(y, oof),
        "selected_epochs": [int(record["selected_epoch"]) for record in folds],
        "folds": folds,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "fold", "resnetd"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "uid": row["uid"],
                "actual": int(y[index]),
                "fold": int(fold_ids[index]),
                "resnetd": float(oof[index]),
            })
    print(json.dumps(summary["oof_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
