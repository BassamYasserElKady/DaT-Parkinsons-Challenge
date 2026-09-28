#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "holdout_900_120_342_3d_ensemble"
SEED = 20260931


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(name, model, x, y, batch_size, epochs, lr, device):
    seed_all(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=True,
    )
    model = model.to(device)
    counts = np.bincount(y, minlength=2).astype(float)
    class_weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(yb)
        scheduler.step()
        print(f"{name} epoch {epoch}/{epochs}: train_loss={total_loss / len(y):.6f}", flush=True)
    return model.eval()


def predict(model, x, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1)), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    values = []
    with torch.inference_mode():
        for (xb,) in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                values.append(torch.softmax(model(xb.to(device, non_blocking=True)), dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(values)


def bounded_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    x = np.load(ROOT / "results" / "downloaded_rigid_adaptive_3d_ensemble" / "input_windows.npy").astype(np.float32, copy=False)
    rows = list(csv.DictReader((ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv").open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    if len(x) != 1362 or len(rows) != 1362:
        raise RuntimeError("Expected 1362 aligned labeled scans")

    all_indices = np.arange(len(y))
    train_val_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=342, random_state=SEED).split(all_indices, y))
    train_rel, val_rel = next(StratifiedShuffleSplit(n_splits=1, test_size=120, random_state=SEED + 1).split(train_val_idx, y[train_val_idx]))
    train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
    assert (len(train_idx), len(val_idx), len(test_idx)) == (900, 120, 342)

    with (OUTPUT / "split.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["uid", "is_pathologic", "partition"])
        writer.writeheader()
        for partition, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
            for index in indices:
                writer.writerow({"uid": rows[index]["uid"], "is_pathologic": int(y[index]), "partition": partition})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = train_model("resnet18", ResNet18CBAM3D(), x[train_idx], y[train_idx], 8, 16, 1e-3, device)
    resnet_val, resnet_test = predict(resnet, x[val_idx], 8, device), predict(resnet, x[test_idx], 8, device)
    del resnet
    torch.cuda.empty_cache()
    densenet = train_model("densenet121", DenseNet121_3D(), x[train_idx], y[train_idx], 2, 14, 3e-4, device)
    densenet_val, densenet_test = predict(densenet, x[val_idx], 2, device), predict(densenet, x[test_idx], 2, device)

    weight = float(minimize_scalar(lambda w: log_loss(y[val_idx], w * resnet_val + (1 - w) * densenet_val, labels=[0, 1]), bounds=(0, 1), method="bounded").x)
    val_raw = weight * resnet_val + (1 - weight) * densenet_val
    test_raw = weight * resnet_test + (1 - weight) * densenet_test
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=SEED).fit(bounded_logit(val_raw).reshape(-1, 1), y[val_idx])
    test_calibrated = calibrator.predict_proba(bounded_logit(test_raw).reshape(-1, 1))[:, 1]

    report = {
        "seed": SEED,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "resnet_weight": weight,
        "densenet_weight": 1 - weight,
        "platt_slope": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "validation_raw": metrics(y[val_idx], val_raw),
        "test_raw": metrics(y[test_idx], test_raw),
        "test_calibrated": metrics(y[test_idx], test_calibrated),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    pd = __import__("pandas")
    pd.DataFrame({"uid": [rows[index]["uid"] for index in test_idx], "is_pathologic": y[test_idx], "resnet_probability": resnet_test, "densenet_probability": densenet_test, "raw_ensemble_probability": test_raw, "calibrated_ensemble_probability": test_calibrated}).to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
