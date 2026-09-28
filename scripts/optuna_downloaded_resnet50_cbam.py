#!/usr/bin/env python3
"""Optuna tuning for the unchanged downloaded ResNet-50 CBAM model."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def builder(path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("downloaded_resnet50_optuna", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_model


def probabilities(model, loader, device):
    model.eval(); truth, score = [], []
    with torch.no_grad():
        for x, y in loader:
            truth.extend(y.numpy().tolist())
            score.extend(torch.softmax(model(x.to(device, non_blocking=True)), 1)[:, 1].cpu().numpy().tolist())
    return np.asarray(truth, dtype=int), np.asarray(score, dtype=float)


def metrics(truth, score):
    predicted = score >= 0.5
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(truth, predicted)), "auroc": float(roc_auc_score(truth, score)),
        "log_loss": float(log_loss(truth, score, labels=[0, 1])),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
    }


def main():
    args = args_parser(); root = args.root.resolve()
    output = root / "results/downloaded_resnet50_cbam_optuna"; output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((root / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open()))
    x = np.load(root / "results/downloaded_resnet50_cbam/input_five_slices.npy").astype(np.float32, copy=False)
    y = np.asarray([int(row["is_pathologic"]) for row in rows], dtype=np.int64)
    split = np.asarray([row["split"] for row in rows]); idx = {p: np.flatnonzero(split == p) for p in ("train", "validation", "test")}
    tensor_x, tensor_y = torch.from_numpy(x), torch.from_numpy(y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_model = builder(root / "original_submission_files/model.py")
    class_counts = np.bincount(y[idx["train"]], minlength=2).astype(float)
    class_weights = torch.tensor(class_counts.sum() / (2 * class_counts), dtype=torch.float32, device=device)

    def fit(params, trial=None, capture=False):
        seed_all(args.seed)
        batch_size = params["batch_size"]
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(TensorDataset(tensor_x[idx["train"]], tensor_y[idx["train"]]), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
        val_loader = DataLoader(TensorDataset(tensor_x[idx["validation"]], tensor_y[idx["validation"]]), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        model = build_model().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=params["label_smoothing"])
        if params["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        elif params["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"], nesterov=True, weight_decay=params["weight_decay"])
        if params["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif params["scheduler"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        else:
            scheduler = None
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                if params["flip_probability"] > 0:
                    mask = torch.rand(len(xb), device=device) < params["flip_probability"]
                    xb[mask] = torch.flip(xb[mask], dims=(-1,))
                if params["noise_std"] > 0:
                    xb = xb + torch.randn_like(xb) * params["noise_std"]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            val_truth, val_score = probabilities(model, val_loader, device)
            val_loss = float(log_loss(val_truth, val_score, labels=[0, 1]))
            if scheduler is not None:
                scheduler.step(val_loss) if params["scheduler"] == "plateau" else scheduler.step()
            if val_loss < best_loss:
                best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
            else:
                stale += 1
            if trial is not None:
                trial.report(best_loss, epoch)
                if trial.should_prune():
                    del model; torch.cuda.empty_cache(); raise optuna.TrialPruned()
            if stale >= 6:
                break
        result = {"validation_log_loss": best_loss, "best_epoch": best_epoch}
        if capture:
            model.load_state_dict(best_state)
            result["model_state_dict"] = best_state
            result["validation_metrics"] = metrics(*probabilities(model, val_loader, device))
        del model; torch.cuda.empty_cache()
        return result

    def objective(trial):
        optimizer = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
        params = {
            "optimizer": optimizer,
            "lr": trial.suggest_float("lr", 1e-5 if optimizer != "SGD" else 1e-4, 3e-3 if optimizer != "SGD" else 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
            "scheduler": trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"]),
            "flip_probability": trial.suggest_categorical("flip_probability", [0.0, 0.25, 0.5]),
            "noise_std": trial.suggest_categorical("noise_std", [0.0, 0.01, 0.03]),
            "momentum": trial.suggest_float("momentum", 0.8, 0.95) if optimizer == "SGD" else 0.9,
        }
        result = fit(params, trial=trial)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        return result["validation_log_loss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4), study_name="downloaded_resnet50_cbam_logloss", storage=f"sqlite:///{output / 'study.db'}", load_if_exists=True)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    best_params = dict(study.best_trial.params); best_params.setdefault("momentum", 0.9)
    final = fit(best_params, capture=True)
    model = build_model().to(device); model.load_state_dict(final["model_state_dict"])
    test_loader = DataLoader(TensorDataset(tensor_x[idx["test"]], tensor_y[idx["test"]]), batch_size=best_params["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_truth, test_score = probabilities(model, test_loader, device); test_metrics = metrics(test_truth, test_score)
    torch.save({"model_state_dict": final["model_state_dict"], "params": best_params, "best_epoch": final["best_epoch"], "validation_metrics": final["validation_metrics"], "test_metrics": test_metrics}, output / "best_model.pt")
    trials = study.trials_dataframe(); trials.to_csv(output / "trials.csv", index=False)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["uid", "actual", "score_patient", "predicted"])
        for pos, index in enumerate(idx["test"]): writer.writerow([rows[index]["uid"], int(test_truth[pos]), float(test_score[pos]), int(test_score[pos] >= 0.5)])
    summary = {"completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials), "pruned_trials": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials), "best_trial": study.best_trial.number, "best_epoch": final["best_epoch"], "best_params": best_params, "validation_metrics": final["validation_metrics"], "test_metrics": test_metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
