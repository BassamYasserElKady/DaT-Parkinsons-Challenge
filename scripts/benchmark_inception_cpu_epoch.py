"""Measure one training epoch for the adaptive-8-slice scratch Inception model."""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cv5_inception_v3_adaptive8_mip as base  # noqa: E402


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results" / "ants_registered_consecutive_resnet18_cbam" / "split.csv").open()))
    labels = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    images = base.build_data(rows)
    train_idx, _ = next(
        StratifiedKFold(5, shuffle=True, random_state=base.SEED).split(images, labels)
    )

    base.seed_all(base.SEED + 1)
    device = torch.device("cpu")
    model = base.make_model().to(device)
    counts = np.bincount(labels[train_idx], minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_x = base.prep(images[train_idx], device)
    train_y = torch.from_numpy(labels[train_idx])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(base.SEED + 1),
        num_workers=0,
    )

    started = time.perf_counter()
    model.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    print(f"cpu_epoch_seconds={elapsed:.2f}", flush=True)
    print(f"cpu_epoch_minutes={elapsed / 60:.2f}", flush=True)
    print(f"train_scans={len(train_idx)} batches={len(loader)}", flush=True)


if __name__ == "__main__":
    main()
