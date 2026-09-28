#!/usr/bin/env python3
"""Test an external EGFR co-expression prior with development-only LOSO CV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spatial_pipeline.adapter import AdapterConfig, DeepSpotCalibrationAdapter, default_device
from spatial_pipeline.manifest import assert_sample_disjoint
from train_adapter_from_tables import read_table
from egfr_rescue_loso import corr_pair, train_global, predict_global


class PriorResidualHead(nn.Module):
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


def batch_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).clamp_min(1e-6)
    return torch.sum(x * y) / denominator


def train_prior_residual(train_ids, samples, egfr_index, module_indices, module_weights, args, device, genes):
    x = np.concatenate([samples[s]["x"] for s in train_ids])
    y = np.concatenate([samples[s]["y"] for s in train_ids])[:, egfr_index]
    module_score = x[:, module_indices] @ module_weights
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(module_score.astype(np.float32))), batch_size=args.batch_size, shuffle=True)
    model = PriorResidualHead(genes, egfr_index).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.residual_lr, weight_decay=args.residual_weight_decay)
    for _ in range(args.epochs):
        model.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            supervised = nn.functional.smooth_l1_loss(prediction, yb)
            prior = 1.0 - batch_correlation(prediction, mb)
            loss = supervised + args.prior_weight * prior
            loss.backward()
            optimizer.step()
    return model


def predict_prior(model, sample, stats, egfr_index, device):
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
    parser.add_argument("--prior", type=Path, default=Path("results/external_priors/egfr/positive_module_genes.csv"))
    parser.add_argument("--module-n", type=int, default=10)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--residual-lr", type=float, default=5e-4)
    parser.add_argument("--residual-weight-decay", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_external_prior_loso"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
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
    prior_frame = pd.read_csv(args.prior if args.prior.is_absolute() else root / args.prior)
    prior_frame = prior_frame[prior_frame["gene"].astype(str).str.upper() != "EGFR"].head(args.module_n)
    module_genes = [str(g).upper() for g in prior_frame["gene"]]
    missing = sorted(set(module_genes) - set(g.upper() for g in genes))
    if missing:
        raise ValueError(f"External module genes missing from fitted panel: {missing}")
    module_indices = np.array([genes.index(g) for g in module_genes], dtype=np.int64)
    module_weights = prior_frame["mean_pearson"].to_numpy(np.float32)
    module_weights = module_weights / np.maximum(module_weights.sum(), 1e-6)

    samples = {}
    for sample_id in development:
        prediction, target = read_table(root, args.tables_dir, sample_id, genes)
        samples[sample_id] = {
            "x": ((prediction - stats["prediction_mean"]) / stats["prediction_std"]).astype(np.float32),
            "y": ((target - stats["target_mean"]) / stats["target_std"]).astype(np.float32),
            "target_raw": target[:, egfr_index].astype(np.float32),
        }
    global_config = AdapterConfig(genes=len(genes), hidden_width=128, latent_width=96, dropout=0.20)
    train_args = SimpleNamespace(epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, weight_decay=1e-4, pearson_weight=0.5, rank_weight=0.15, rank_pairs=3000, residual_lr=args.residual_lr, residual_weight_decay=args.residual_weight_decay)

    rows = []
    for fold, validation_id in enumerate(sorted(development), start=1):
        train_ids = [sample_id for sample_id in development if sample_id != validation_id]
        assert_sample_disjoint(train_ids, [validation_id])
        global_model = train_global(train_ids, samples, global_config, train_args, device)
        prior_model = train_prior_residual(train_ids, samples, egfr_index, module_indices, module_weights, args, device, len(genes))
        sample = samples[validation_id]
        target = sample["target_raw"]
        raw = sample["x"][:, egfr_index] * stats["target_std"][egfr_index] + stats["target_mean"][egfr_index]
        global_prediction = predict_global(global_model, sample, stats, device)[:, egfr_index]
        prior_prediction = predict_prior(prior_model, sample, stats, egfr_index, device)
        for model_name, prediction in (("raw", raw), ("global_adapter", global_prediction), ("external_prior_residual", prior_prediction)):
            pearson, spearman = corr_pair(prediction, target)
            rows.append({"fold": fold, "held_out_sample": validation_id, "model": model_name, "tiles": len(target), "spatial_pearson": pearson, "spatial_spearman": spearman})
        print(f"completed fold {fold}/{len(development)}: held_out={validation_id}", flush=True)

    result = pd.DataFrame(rows)
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "loso_metrics.csv", index=False)
    summary = result.groupby("model", as_index=False)[["spatial_pearson", "spatial_spearman"]].mean()
    summary.to_csv(output / "loso_mean_metrics.csv", index=False)
    pd.DataFrame({"gene": module_genes, "weight": module_weights}).to_csv(output / "module_used.csv", index=False)
    report = [
        "# EGFR external-prior leave-one-development-sample-out evaluation",
        "",
        f"External module: `{', '.join(module_genes)}`.",
        "",
        f"Development samples: `{len(development)}`. Final-test samples were not loaded or evaluated: `{','.join(final_test)}`.",
        "",
        summary.to_markdown(index=False),
        "",
        "The external datasets determine only a population-level co-expression regularizer. They are not patient-specific conditioning inputs.",
    ]
    (output / "loso_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))
    print(f"output={output}; device={device}")


if __name__ == "__main__":
    main()
