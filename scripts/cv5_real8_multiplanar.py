#!/usr/bin/env python3
"""Five-fold 8-slice ensemble using axial, coronal, and sagittal input channels."""

from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from cv5_original_flip_vs_dml import DenseNet121_3D, ResNet18CBAM3D, seed_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cv5_real8_multiplanar"
REGISTERED = ROOT / "results/ants_registration_mni"
ROWS_PATH = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
SEED = 20260931
SIGMA = 0.75
NATIVE_DEPTH = 8
TARGET_SHAPE = (16, 64, 64)


def preprocess_one(payload):
    index, uid, label = payload
    import sys
    localizer_dir = str(ROOT / "original_submission_files")
    if localizer_dir not in sys.path:
        sys.path.insert(0, localizer_dir)
    from adaptive_localizer import adaptive_localization
    volume = np.asarray(nib.load(str(REGISTERED / f"{uid}.nii.gz")).get_fdata(), dtype=np.float32)
    volume = np.where(np.isfinite(volume), np.maximum(volume, 0), 0)
    volume = ndimage.gaussian_filter(volume, sigma=SIGMA, mode="nearest")
    best, _ = adaptive_localization(volume)
    zc = (int(best["start"]) + int(best["end"])) // 2
    hot = volume >= np.percentile(volume[volume > 0], 99.75)
    coords = np.argwhere(hot)
    yc = int(np.median(coords[:, 1])) if len(coords) else volume.shape[1] // 2
    xc = int(np.median(coords[:, 0])) if len(coords) else volume.shape[0] // 2
    centers = [xc, yc, zc]
    planes = []
    for axis, center in enumerate(centers):
        start = min(max(center - NATIVE_DEPTH // 2, 0), volume.shape[axis] - NATIVE_DEPTH)
        indices = range(start, start + NATIVE_DEPTH)
        if axis == 2:  # axial: z slices, presented as y by x
            plane_stack = np.stack([volume[:, :, i].T for i in indices], axis=0)
        elif axis == 1:  # coronal: y slices, presented as z by x
            plane_stack = np.stack([volume[:, i, :].T for i in indices], axis=0)
        else:  # sagittal: x slices, presented as z by y
            plane_stack = np.stack([volume[i, :, :].T for i in indices], axis=0)
        if plane_stack.shape[1:] != (109, 91):
            plane_stack = ndimage.zoom(plane_stack, (1, 109 / plane_stack.shape[1], 91 / plane_stack.shape[2]), order=1)
        planes.append(plane_stack)
    cube = np.stack(planes, axis=0)
    positive = cube[cube > 0]
    if positive.size == 0 or float(positive.std()) < 1e-8:
        raise RuntimeError(f"invalid positive voxels for {uid}")
    cube[cube > 0] = (cube[cube > 0] - float(positive.mean())) / float(positive.std())
    if cube.shape != (3, 8, 109, 91) or not np.all(np.isfinite(cube)):
        raise RuntimeError(f"invalid cube for {uid}: {cube.shape}")
    return index, uid, label, cube.astype(np.float32), zc, yc, xc


def build_inputs(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "multiplanar_native.npy"
    if path.exists():
        x = np.load(path).astype(np.float32, copy=False)
        print(f"reusing_inputs={path} shape={x.shape}", flush=True)
        return x
    x = np.empty((len(rows), 3, NATIVE_DEPTH, 109, 91), dtype=np.float32)
    audit = [None] * len(rows)
    payloads = [(i, row["uid"], int(row["is_pathologic"])) for i, row in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(preprocess_one, payload) for payload in payloads]
        for count, future in enumerate(as_completed(futures), 1):
            index, uid, label, cube, zc, yc, xc = future.result()
            x[index] = cube
            audit[index] = {"uid": uid, "is_pathologic": label, "axial_center": zc, "coronal_center": yc, "sagittal_center": xc}
            if count % 100 == 0 or count == len(rows):
                print(f"preprocessed={count}/{len(rows)}", flush=True)
    np.save(path, x)
    with (OUT / "slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    return x


def resize_standardize(native, train_index):
    means = native[train_index].mean(axis=(0, 2, 3, 4), dtype=np.float64).astype(np.float32)
    stds = native[train_index].std(axis=(0, 2, 3, 4), dtype=np.float64).astype(np.float32)
    output = np.empty((len(native), 3, *TARGET_SHAPE), dtype=np.float32)
    zoom = (TARGET_SHAPE[0] / 8, TARGET_SHAPE[1] / 109, TARGET_SHAPE[2] / 91)
    for i in range(len(native)):
        standardized = (native[i] - means[:, None, None, None]) / stds[:, None, None, None]
        for channel in range(3):
            output[i, channel] = ndimage.zoom(standardized[channel], zoom, order=1).astype(np.float32)
    return output


def make_models():
    resnet = ResNet18CBAM3D()
    resnet.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    dense = DenseNet121_3D()
    dense.features._modules["conv0"] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    return resnet, dense


def class_loss(y, ids, device):
    counts = np.bincount(y[ids], minlength=2).astype(float)
    return nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))


def train(model, x, y, ids, batch, epochs, lr, device, seed, name):
    seed_all(seed); model = model.to(device)
    ds = TensorDataset(torch.from_numpy(x[ids]), torch.from_numpy(y[ids]))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed), pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = class_loss(y, ids, device)
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            if torch.rand(()) < 0.5: xb = xb.flip(-1)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = ce(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += float(loss.detach()) * len(yb)
        sched.step(); print(f"{name} epoch={epoch}/{epochs} loss={total / len(ids):.6f}", flush=True)
    return model.eval()


def predict(model, x, ids, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x[ids])), batch_size=4, shuffle=False, pin_memory=True)
    out = []; model.eval()
    with torch.inference_mode():
        for (xb,) in loader:
            out.append(torch.softmax(model(xb.to(device, non_blocking=True)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    p = np.clip(np.asarray(p), 0.005, 0.995)
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "accuracy": float(accuracy_score(y, p >= 0.5)), "incorrect": int(np.sum((p >= 0.5) != y))}


def main():
    OUT.mkdir(parents=True, exist_ok=True); seed_all(SEED)
    rows = list(csv.DictReader(ROWS_PATH.open(newline="", encoding="utf-8"))); y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    native = build_inputs(rows); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device} sigma={SIGMA}", flush=True)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(native, y)); fold_id = np.empty(len(y), dtype=np.int64)
    ro = np.full(len(y), np.nan); do = np.full(len(y), np.nan); folds = []
    for fold, (train_index, outer_index) in enumerate(splits, 1):
        fold_id[outer_index] = fold; x = resize_standardize(native, train_index); print(f"fold={fold}/5 train={len(train_index)} outer={len(outer_index)}", flush=True)
        r, d = make_models()
        r = train(r, x, y, train_index, 8, 16, 1e-3, device, SEED + fold, f"multiplanar_resnet_f{fold}"); ro[outer_index] = predict(r, x, outer_index, device); del r; torch.cuda.empty_cache()
        d = train(d, x, y, train_index, 2, 14, 3e-4, device, SEED + fold, f"multiplanar_dense_f{fold}"); do[outer_index] = predict(d, x, outer_index, device); del d, x; torch.cuda.empty_cache()
        ep = 0.5 * ro[outer_index] + 0.5 * do[outer_index]; record = {"fold": fold, "train": len(train_index), "outer": len(outer_index), "resnet": metrics(y[outer_index], ro[outer_index]), "densenet": metrics(y[outer_index], do[outer_index]), "equal_ensemble": metrics(y[outer_index], ep)}; folds.append(record)
        (OUT / "progress.json").write_text(json.dumps(folds, indent=2) + "\n"); np.savez(OUT / "oof_progress.npz", resnet=ro, densenet=do, fold=fold_id); print(json.dumps(record), flush=True)
    ep = 0.5 * ro + 0.5 * do; clipped = np.clip(ep, 0.005, 0.995); summary = {"n_scans": len(y), "input": "three channels: axial, coronal, sagittal 8-slice slabs resized to depth 16", "gaussian_sigma_voxels": SIGMA, "training_augmentation": "horizontal flip only", "ensemble_weighting": "0.5 ResNet + 0.5 DenseNet", "resnet_oof": metrics(y, ro), "densenet_oof": metrics(y, do), "equal_ensemble_oof": metrics(y, ep), "equal_ensemble_clipped_005_995_oof": metrics(y, clipped), "folds": folds}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "oof_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "fold", "resnet", "densenet", "equal_ensemble", "equal_ensemble_clipped"]); writer.writeheader()
        for i, row in enumerate(rows): writer.writerow({"uid": row["uid"], "actual": int(y[i]), "fold": int(fold_id[i]), "resnet": float(ro[i]), "densenet": float(do[i]), "equal_ensemble": float(ep[i]), "equal_ensemble_clipped": float(clipped[i])})
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("oof")}, indent=2), flush=True)


if __name__ == "__main__": main()
