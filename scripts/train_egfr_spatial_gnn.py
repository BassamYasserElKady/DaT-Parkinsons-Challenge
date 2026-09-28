#!/usr/bin/env python3
"""Train a sample-disjoint spatial EGFR predictor.

Inputs are the existing per-sample DeepSpot/Xenium parquet tables. Tiles are
nodes in a grid graph, with edges between adjacent tiles. Development samples
are used for grouped validation and training; final-test samples are evaluated
only after the model is frozen.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_pipeline.egfr_spatial_model import EGFRSpatialConfig, EGFRSpatialGraphNet, egfr_spatial_loss
from spatial_pipeline.manifest import assert_sample_disjoint


@dataclass
class GraphSample:
    sample_id: str
    x: torch.Tensor
    coordinates: torch.Tensor
    edge_index: torch.Tensor
    target: torch.Tensor
    target_raw: torch.Tensor

    def on(self, device: torch.device) -> "GraphSample":
        return GraphSample(
            self.sample_id,
            self.x.to(device),
            self.coordinates.to(device),
            self.edge_index.to(device),
            self.target.to(device),
            self.target_raw.to(device),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/deepspot_tables"))
    parser.add_argument("--stats", type=Path, default=Path("results/deepspot_tables/development_normalization.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/egfr_spatial_gnn"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-egfr-input", action="store_true", help="remove the raw DeepSpot EGFR feature from inputs")
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def grid_edges(xy: np.ndarray) -> np.ndarray:
    grid = np.rint(xy / 224.0).astype(np.int64)
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(grid)}
    edges: list[tuple[int, int]] = []
    for i, (x, y) in enumerate(grid):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            j = lookup.get((int(x + dx), int(y + dy)))
            if j is not None:
                edges.append((i, j))
                edges.append((j, i))
    if not edges:
        # Self edges keep the layer well-defined for a degenerate input.
        edges = [(i, i) for i in range(len(xy))]
    return np.asarray(edges, dtype=np.int64).T


def read_sample(root: Path, tables_dir: Path, sample_id: str, genes: list[str], pred_mean: np.ndarray, pred_std: np.ndarray, target_mean: float, target_std: float, exclude_egfr: bool) -> GraphSample:
    sample_dir = resolve(root, tables_dir) / sample_id
    prediction = pd.read_parquet(next(sample_dir.glob("*_deepspot_scgpt_predictions.parquet")))
    target = pd.read_parquet(next(sample_dir.glob("*_xenium_log1p_cp10k_targets.parquet")))
    if not prediction[["spot", "x", "y"]].equals(target[["spot", "x", "y"]]):
        raise ValueError(f"{sample_id}: prediction/target coordinates do not match")
    missing = [g for g in genes if g not in prediction.columns or g not in target.columns]
    if missing:
        raise ValueError(f"{sample_id}: missing genes {missing[:5]}")
    values = prediction[genes].to_numpy(np.float32)
    inputs = (values - pred_mean) / pred_std
    input_genes = genes
    if exclude_egfr:
        keep = np.asarray([g != "EGFR" for g in genes])
        inputs = inputs[:, keep]
        input_genes = [g for g in genes if g != "EGFR"]
    xy = prediction[["x", "y"]].to_numpy(np.float32)
    grid_xy = xy / 224.0
    scale = np.maximum(grid_xy.max(axis=0), 1.0)
    coordinates = grid_xy / scale
    target_raw = target["EGFR"].to_numpy(np.float32)
    target_standardized = (target_raw - target_mean) / target_std
    return GraphSample(
        sample_id=sample_id,
        x=torch.from_numpy(inputs),
        coordinates=torch.from_numpy(coordinates.astype(np.float32)),
        edge_index=torch.from_numpy(grid_edges(xy)),
        target=torch.from_numpy(target_standardized),
        target_raw=torch.from_numpy(target_raw),
    )


def grouped_folds(sample_ids: list[str], n_folds: int, seed: int):
    if n_folds < 2 or n_folds > len(sample_ids):
        raise ValueError("validation-folds must be between 2 and the number of development samples")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(sorted(sample_ids), dtype=object)[rng.permutation(len(sample_ids))])
    folds = []
    for validation in np.array_split(shuffled, n_folds):
        validation_ids = [str(v) for v in validation]
        training_ids = [sid for sid in sample_ids if sid not in validation_ids]
        assert_sample_disjoint(training_ids, validation_ids)
        folds.append((training_ids, validation_ids))
    return folds


def safe_corr(pred: np.ndarray, target: np.ndarray, method: str) -> float:
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return float("nan")
    result = pearsonr(pred, target) if method == "pearson" else spearmanr(pred, target)
    return float(result.statistic)


def binary_auc(score: np.ndarray, label: np.ndarray) -> float:
    positive = label.astype(bool)
    n_pos = int(positive.sum()); n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def predict(model: EGFRSpatialGraphNet, sample: GraphSample, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        output = model(sample.x.to(device), sample.coordinates.to(device), sample.edge_index.to(device))
        abundance = output["abundance"].cpu().numpy()
        presence = torch.sigmoid(output["presence"]).cpu().numpy()
    return abundance, presence


def evaluate(model: EGFRSpatialGraphNet, ids: list[str], samples: dict[str, GraphSample], target_mean: float, target_std: float, device: torch.device, output_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    all_pred=[]; all_target=[]; all_prob=[]; all_label=[]
    for sid in ids:
        pred_std, prob = predict(model, samples[sid], device)
        target_raw = samples[sid].target_raw.cpu().numpy()
        pred_raw = pred_std * target_std + target_mean
        label = target_raw > 0
        rows.append({
            "sample_id": sid,
            "n_tiles": len(target_raw),
            "pearson": safe_corr(pred_raw, target_raw, "pearson"),
            "spearman": safe_corr(pred_raw, target_raw, "spearman"),
            "egfr_detection_auroc": binary_auc(prob, label),
            "target_positive_fraction": float(label.mean()),
            "prediction_positive_fraction": float((prob >= 0.5).mean()),
        })
        all_pred.append(pred_raw); all_target.append(target_raw); all_prob.append(prob); all_label.append(label)
        if output_dir is not None:
            frame = pd.DataFrame({"x": samples[sid].coordinates[:, 0].cpu().numpy(), "y": samples[sid].coordinates[:, 1].cpu().numpy(), "predicted_egfr": pred_raw, "measured_egfr": target_raw, "egfr_detection_probability": prob})
            frame.to_parquet(output_dir / f"{sid}_predictions.parquet", index=False)
    pooled_pred=np.concatenate(all_pred); pooled_target=np.concatenate(all_target); pooled_prob=np.concatenate(all_prob); pooled_label=np.concatenate(all_label)
    rows.append({
        "sample_id":"POOLED",
        "n_tiles":len(pooled_target),
        "pearson":safe_corr(pooled_pred, pooled_target, "pearson"),
        "spearman":safe_corr(pooled_pred, pooled_target, "spearman"),
        "egfr_detection_auroc":binary_auc(pooled_prob, pooled_label),
        "target_positive_fraction":float(pooled_label.mean()),
        "prediction_positive_fraction":float((pooled_prob >= 0.5).mean()),
    })
    return pd.DataFrame(rows)


def fit(train_ids: list[str], samples: dict[str, GraphSample], config: EGFRSpatialConfig, args: argparse.Namespace, device: torch.device, epochs: int | None = None) -> EGFRSpatialGraphNet:
    model=EGFRSpatialGraphNet(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_samples=[samples[sid].on(device) for sid in train_ids]
    for epoch in range(epochs or args.epochs):
        model.train(); order=np.random.permutation(len(train_samples)); losses=[]
        for idx in order:
            sample=train_samples[int(idx)]
            optimizer.zero_grad(set_to_none=True)
            output=model(sample.x, sample.coordinates, sample.edge_index)
            loss=egfr_spatial_loss(output, sample.target, (sample.target_raw > 0).float())
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss["total"].detach().cpu()))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch+1} train_loss={np.mean(losses):.4f}", flush=True)
    return model


def main() -> None:
    args=parse_args()
    if args.epochs <= 0: raise ValueError("epochs must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=args.root.resolve()
    manifest=pd.read_csv(resolve(root,args.manifest))
    development=manifest.loc[manifest["split"].eq("development"),"sample_id"].astype(str).tolist()
    final_test=manifest.loc[manifest["split"].eq("final_test"),"sample_id"].astype(str).tolist()
    assert_sample_disjoint(development,final_test)
    with np.load(resolve(root,args.stats),allow_pickle=False) as stats:
        genes=[str(g) for g in stats["genes"].tolist()]
        pred_mean=stats["prediction_mean"].astype(np.float32); pred_std=stats["prediction_std"].astype(np.float32)
        target_mean_all=stats["target_mean"].astype(np.float32); target_std_all=stats["target_std"].astype(np.float32)
    egfr_idx=genes.index("EGFR")
    target_mean=float(target_mean_all[egfr_idx]); target_std=float(target_std_all[egfr_idx])
    samples={sid:read_sample(root,args.tables_dir,sid,genes,pred_mean,pred_std,target_mean,target_std,args.exclude_egfr_input) for sid in development+final_test}
    input_genes=[g for g in genes if not(args.exclude_egfr_input and g=="EGFR")]
    config=EGFRSpatialConfig(input_genes=len(input_genes),hidden_width=args.hidden_width,dropout=args.dropout)
    output_dir=resolve(root,args.output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    fold_rows=[]
    for fold,(train_ids,val_ids) in enumerate(grouped_folds(development,args.validation_folds,args.seed),1):
        print(f"fold={fold} train={','.join(sorted(train_ids))} validation={','.join(sorted(val_ids))}",flush=True)
        model=fit(train_ids,samples,config,args,device)
        metrics=evaluate(model,val_ids,samples,target_mean,target_std,device)
        metrics.insert(0,"fold",fold); fold_rows.append(metrics)
    validation=pd.concat(fold_rows,ignore_index=True)
    final_model=fit(development,samples,config,args,device)
    final_metrics=evaluate(final_model,final_test,samples,target_mean,target_std,device,output_dir)
    validation.to_csv(output_dir/"grouped_validation_metrics.csv",index=False)
    final_metrics.to_csv(output_dir/"final_test_metrics.csv",index=False)
    torch.save({"model_state_dict":final_model.state_dict(),"config":vars(config),"genes":genes,"input_genes":input_genes,"target_mean":target_mean,"target_std":target_std,"development_samples":development,"final_test_samples":final_test,"exclude_egfr_input":args.exclude_egfr_input},output_dir/"egfr_spatial_gnn_final.pt")
    print(f"device={device}; input_genes={len(input_genes)}; development={len(development)}; final_test={len(final_test)}")
    print("Grouped validation:"); print(validation.to_string(index=False))
    print("Final test:"); print(final_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
