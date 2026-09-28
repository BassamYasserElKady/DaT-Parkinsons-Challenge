#!/usr/bin/env python3
"""Train final 3D ensemble components on all labeled scans."""
from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/full_data_calibrated_3d_ensemble_flip_clip"
SEED = 20260931

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def train(name, model, x, y, batch_size, epochs, lr, device):
    seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    data = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    model = model.to(device)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in data:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"): loss = criterion(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += float(loss.detach()) * len(yb)
        scheduler.step(); row = {"epoch": epoch, "loss": total / len(y), "learning_rate": optimizer.param_groups[0]["lr"]}; history.append(row)
        print(f"model={name} epoch={epoch}/{epochs} loss={row['loss']:.6f} lr={row['learning_rate']:.8f}", flush=True)
    return model.cpu().state_dict(), history

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_all(SEED)
    x = np.load(ROOT / "results/downloaded_rigid_adaptive_3d_ensemble/input_windows.npy").astype(np.float32, copy=False)
    rows = list(csv.DictReader((ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    if len(x) != 1362 or len(y) != 1362: raise RuntimeError("expected 1362 training scans")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rstate, rhistory = train("resnet18", ResNet18CBAM3D(), x, y, 8, 16, 1e-3, device)
    torch.cuda.empty_cache()
    dstate, dhistory = train("densenet121", DenseNet121_3D(), x, y, 2, 14, 3e-4, device)
    fold_weights = [0.1040855061449809, 0.5783490912163899, 0.518656303278787, 0.6594996245701895, 0.44561961100525854]
    fold_slopes = [0.9826147163764261, 0.9808237759296075, 1.6712186129877702, 2.0129506524704857, 0.9051458268062743]
    fold_intercepts = [0.13234131238715605, 0.09321890353394849, -0.2024147240508331, 0.010778904664923818, 0.3670293772029915]
    metadata = {"seed": SEED, "training_scan_count": len(y), "input_shape": [1,16,64,64], "resnet_epochs": 16, "densenet_epochs": 14, "resnet_weight": float(np.median(fold_weights)), "densenet_weight": float(1-np.median(fold_weights)), "platt_slope": float(np.median(fold_slopes)), "platt_intercept": float(np.median(fold_intercepts))}
    torch.save({"resnet_state_dict": rstate, "densenet_state_dict": dstate, **metadata}, OUTPUT / "ensemble_weights.pt")
    (OUTPUT / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, history in (("resnet_history.csv", rhistory), ("densenet_history.csv", dhistory)):
        with (OUTPUT / name).open("w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print(json.dumps(metadata, indent=2), flush=True)

if __name__ == "__main__": main()
