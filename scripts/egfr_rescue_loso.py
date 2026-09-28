#!/usr/bin/env python3
"""Development-only leave-one-sample-out EGFR rescue experiment.

Compares the existing global calibration adapter with a small, regularized
EGFR-specific residual head. Final-test samples are not loaded or evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device
from spatial_pipeline.manifest import assert_sample_disjoint
from spatial_pipeline.losses import calibration_loss
from train_adapter_from_tables import read_table


class EGFRResidualHead(nn.Module):
    """Low-capacity residual correction for one gene, initialized at raw output."""

    def __init__(self, genes: int, egfr_index: int, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.egfr_index = egfr_index
        self.correction = nn.Sequential(
            nn.Linear(genes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.egfr_index] + self.correction(x).squeeze(1)


def corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1]), float(spearmanr(x[valid], y[valid]).statistic)


def train_global(train_ids, samples, config, args, device):
    x = np.concatenate([samples[s]["x"] for s in train_ids])
    y = np.concatenate([samples[s]["y"] for s in train_ids])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=args.batch_size, shuffle=True)
    model = DeepSpotCalibrationAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = calibration_loss(model(xb), yb, pearson_weight=args.pearson_weight, rank_weight=args.rank_weight, rank_pairs=args.rank_pairs)
            losses["total"].backward()
            optimizer.step()
    return model


def train_residual(train_ids, samples, egfr_index, args, device, genes):
    x = np.concatenate([samples[s]["x"] for s in train_ids])
    y = np.concatenate([samples[s]["y"] for s in train_ids])[:, egfr_index]
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=args.batch_size, shuffle=True)
    model = EGFRResidualHead(genes, egfr_index).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.residual_lr, weight_decay=args.residual_weight_decay)
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def predict_global(model, sample, stats, device):
    model.eval()
    with torch.inference_mode():
        z = model(torch.from_numpy(sample["x"]).to(device)).cpu().numpy()
    return z * stats["target_std"] + stats["target_mean"]


def predict_residual(model, sample, stats, egfr_index, device):
    model.eval()
    with torch.inference_mode():
        z = model(torch.from_numpy(sample["x"]).to(device)).cpu().numpy()
    return z * stats["target_std"][egfr_index] + stats["target_mean"][egfr_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_rescue_loso"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-weight-decay", type=float, default=1e-3)
    parser.add_argument("--pearson-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.15)
    parser.add_argument("--rank-pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    root = args.root.resolve()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = default_device()

    manifest = pd.read_csv(args.manifest if args.manifest.is_absolute() else root / args.manifest)
    development = manifest.loc[manifest["split"] == "development", "sample_id"].astype(str).tolist()
    final_test = manifest.loc[manifest["split"] == "final_test", "sample_id"].astype(str).tolist()
    assert_sample_disjoint(development, final_test)
    with np.load(args.stats if args.stats.is_absolute() else root / args.stats, allow_pickle=False) as stats_file:
        genes = [str(g) for g in stats_file["genes"].tolist()]
        stats = {key: stats_file[key].astype(np.float32) for key in ("prediction_mean", "prediction_std", "target_mean", "target_std")}
    egfr_index = genes.index("EGFR")
    samples = {}
    for sample_id in development:
        prediction, target = read_table(root, args.tables_dir, sample_id, genes)
        samples[sample_id] = {
            "x": ((prediction - stats["prediction_mean"]) / stats["prediction_std"]).astype(np.float32),
            "y": ((target - stats["target_mean"]) / stats["target_std"]).astype(np.float32),
            "target_raw": target[:, egfr_index].astype(np.float32),
        }

    global_config = AdapterConfig(genes=len(genes), hidden_width=128, latent_width=96, dropout=0.20)
    train_args = SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pearson_weight=args.pearson_weight,
        rank_weight=args.rank_weight,
        rank_pairs=args.rank_pairs,
    )
    rows = []
    for fold, validation_id in enumerate(sorted(development), start=1):
        train_ids = [sample_id for sample_id in development if sample_id != validation_id]
        assert_sample_disjoint(train_ids, [validation_id])
        global_model = train_global(train_ids, samples, global_config, train_args, device)
        residual_model = train_residual(train_ids, samples, egfr_index, args, device, len(genes))
        sample = samples[validation_id]
        raw = sample["x"][:, egfr_index] * stats["target_std"][egfr_index] + stats["target_mean"][egfr_index]
        target = sample["target_raw"]
        global_pred = predict_global(global_model, sample, stats, device)[:, egfr_index]
        residual_pred = predict_residual(residual_model, sample, stats, egfr_index, device)
        for model_name, prediction in (("raw", raw), ("global_adapter", global_pred), ("egfr_residual", residual_pred)):
            pearson, spearman = corr_pair(prediction, target)
            rows.append({"fold": fold, "held_out_sample": validation_id, "model": model_name, "tiles": len(target), "spatial_pearson": pearson, "spatial_spearman": spearman})
        print(f"completed fold {fold}/{len(development)}: held_out={validation_id}", flush=True)

    result = pd.DataFrame(rows)
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "loso_metrics.csv", index=False)
    summary = result.groupby("model", as_index=False)[["spatial_pearson", "spatial_spearman"]].mean()
    summary.to_csv(output / "loso_mean_metrics.csv", index=False)
    report = [
        "# EGFR rescue leave-one-development-sample-out evaluation",
        "",
        f"Development samples: `{len(development)}`. Final-test samples were not loaded or evaluated: `{','.join(final_test)}`.",
        "",
        "The EGFR residual head is a small regularized correction initialized at the raw DeepSpot-M EGFR output. Model selection must use these held-out development metrics only.",
        "",
        summary.to_markdown(index=False),
    ]
    (output / "loso_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))
    print(f"output={output}; device={device}")


if __name__ == "__main__":
    main()
