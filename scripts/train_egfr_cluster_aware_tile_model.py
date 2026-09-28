#!/usr/bin/env python3
"""Benchmark a leakage-safe, cluster-aware Xenium EGFR tile model.

The unsupervised tile clusters are learned from DeepSpot-predicted genes with
EGFR removed.  Xenium EGFR is used only as the regression target, never to
construct the clusters or neighborhood features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "results" / "deepspot_tables"
MANIFEST_PATH = ROOT / "data" / "sample_manifest.csv"
OUT_ROOT = ROOT / "results" / "egfr_cluster_aware_tile_model"


@dataclass
class SampleData:
    sample: str
    x: np.ndarray
    y: np.ndarray
    coords: np.ndarray


@dataclass
class ClusterModel:
    gene_scaler: StandardScaler
    pca: PCA
    kmeans: MiniBatchKMeans
    egfr_index: int
    n_clusters: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=17)
    return parser.parse_args()


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    if "sample" not in manifest.columns and "sample_id" in manifest.columns:
        manifest = manifest.rename(columns={"sample_id": "sample"})
    required = {"sample", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return manifest


def load_sample(sample: str, genes: list[str]) -> SampleData:
    pred_path = TABLE_ROOT / sample / f"{sample.lower()}_deepspot_scgpt_predictions.parquet"
    target_path = TABLE_ROOT / sample / f"{sample.lower()}_xenium_log1p_cp10k_targets.parquet"
    pred = pd.read_parquet(pred_path)
    target = pd.read_parquet(target_path)

    if list(pred.columns[:3]) != ["spot", "x", "y"]:
        raise ValueError(f"Unexpected prediction columns for {sample}: {pred.columns[:3].tolist()}")
    if list(target.columns[:3]) != ["spot", "x", "y"]:
        raise ValueError(f"Unexpected target columns for {sample}: {target.columns[:3].tolist()}")
    if pred.shape[0] != target.shape[0] or not np.array_equal(
        pred[["spot", "x", "y"]].to_numpy(), target[["spot", "x", "y"]].to_numpy()
    ):
        raise ValueError(f"Prediction/target row alignment failed for {sample}")

    missing = [gene for gene in genes if gene not in pred.columns]
    if missing:
        raise ValueError(f"{sample} is missing predicted genes: {missing[:5]}")
    if "EGFR" not in target.columns:
        raise ValueError(f"{sample} target table has no EGFR column")

    x = pred[genes].to_numpy(dtype=np.float32)
    y = target["EGFR"].to_numpy(dtype=np.float32)
    coords = pred[["x", "y"]].to_numpy(dtype=np.float32)
    return SampleData(sample=sample, x=x, y=y, coords=coords)


def fit_cluster_model(
    samples: list[SampleData],
    genes: list[str],
    n_clusters: int,
    pca_components: int,
    random_state: int,
) -> ClusterModel:
    egfr_index = genes.index("EGFR")
    keep = np.ones(len(genes), dtype=bool)
    keep[egfr_index] = False
    profiles = np.concatenate([sample.x[:, keep] for sample in samples], axis=0)

    gene_scaler = StandardScaler().fit(profiles)
    scaled = gene_scaler.transform(profiles)
    n_components = min(pca_components, scaled.shape[1], scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=random_state)
    latent = pca.fit_transform(scaled)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init=10,
        max_iter=100,
    ).fit(latent)
    return ClusterModel(
        gene_scaler=gene_scaler,
        pca=pca,
        kmeans=kmeans,
        egfr_index=egfr_index,
        n_clusters=n_clusters,
    )


def add_cluster_features(sample: SampleData, cluster_model: ClusterModel) -> np.ndarray:
    keep = np.ones(sample.x.shape[1], dtype=bool)
    keep[cluster_model.egfr_index] = False
    latent = cluster_model.pca.transform(
        cluster_model.gene_scaler.transform(sample.x[:, keep])
    )
    labels = cluster_model.kmeans.predict(latent)

    current = np.eye(cluster_model.n_clusters, dtype=np.float32)[labels]
    if sample.x.shape[0] <= 1:
        neighbor_fraction = np.zeros_like(current)
    else:
        # Tile coordinates form an approximately regular grid.  The nearest
        # eight tiles provide a simple, resolution-matched local composition.
        tree = cKDTree(sample.coords.astype(np.float64))
        n_neighbors = min(9, sample.x.shape[0])
        _, indices = tree.query(sample.coords.astype(np.float64), k=n_neighbors)
        neighbor_labels = labels[indices[:, 1:]]
        neighbor_fraction = np.eye(cluster_model.n_clusters, dtype=np.float32)[
            neighbor_labels
        ].mean(axis=1)

    return np.concatenate([sample.x, current, neighbor_fraction], axis=1).astype(np.float32)


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(scaler.transform(train_x), train_y)
    return scaler, model


def predict(scaler: StandardScaler, model: Ridge, x: np.ndarray) -> np.ndarray:
    # Expression is nonnegative; clipping is applied consistently to both
    # baseline and cluster-aware predictions.
    return np.maximum(model.predict(scaler.transform(x)), 0.0).astype(np.float32)


def metric_row(sample: str, group: str, method: str, y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    row = {
        "sample": sample,
        "group": group,
        "method": method,
        "n_tiles": int(y.size),
        "mean_target": float(y.mean()),
        "mean_prediction": float(pred.mean()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
    }
    if np.unique(y).size > 1 and np.unique(pred).size > 1:
        row["pearson_r"] = float(pearsonr(y, pred).statistic)
        row["spearman_r"] = float(spearmanr(y, pred).statistic)
    else:
        row["pearson_r"] = np.nan
        row["spearman_r"] = np.nan
    return row


def concatenate(samples: list[SampleData], features: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(
        [features[sample.sample] if features is not None else sample.x for sample in samples],
        axis=0,
    )
    y = np.concatenate([sample.y for sample in samples], axis=0)
    return x, y


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    development_names = manifest.loc[manifest["split"].eq("development"), "sample"].tolist()
    final_names = manifest.loc[manifest["split"].eq("final_test"), "sample"].tolist()
    all_names = development_names + final_names
    if not development_names or not final_names:
        raise ValueError("Manifest must define both development and final_test samples")

    first_pred = pd.read_parquet(
        TABLE_ROOT / development_names[0] / f"{development_names[0].lower()}_deepspot_scgpt_predictions.parquet"
    )
    genes = list(first_pred.columns[3:])
    if "EGFR" not in genes:
        raise ValueError("DeepSpot predictions have no EGFR feature")

    samples = {sample: load_sample(sample, genes) for sample in all_names}
    development = [samples[name] for name in development_names]
    final_test = [samples[name] for name in final_names]

    # Existing manifest folds define the sample-level validation protocol.
    fold_names = manifest.loc[manifest["split"].eq("development"), ["sample"]].copy()
    # The manifest has no predefined fold column, so use leave-one-sample-out
    # validation on development samples.
    fold_names["fold"] = np.arange(len(fold_names))
    fold_names["fold"] = fold_names["fold"].fillna(-1).astype(int)

    fold_rows: list[dict] = []
    fold_cluster_metadata: list[dict] = []
    for fold in sorted(fold_names["fold"].unique()):
        validation_names = fold_names.loc[fold_names["fold"].eq(fold), "sample"].tolist()
        training_names = [name for name in development_names if name not in validation_names]
        if not training_names or not validation_names:
            continue
        train_samples = [samples[name] for name in training_names]
        val_samples = [samples[name] for name in validation_names]
        cluster_model = fit_cluster_model(
            train_samples,
            genes,
            args.n_clusters,
            args.pca_components,
            args.random_state + fold,
        )
        train_aug = {sample.sample: add_cluster_features(sample, cluster_model) for sample in train_samples}
        val_aug = {sample.sample: add_cluster_features(sample, cluster_model) for sample in val_samples}
        train_x, train_y = concatenate(train_samples)
        train_aug_x, _ = concatenate(train_samples, train_aug)

        baseline_scaler, baseline_model = fit_ridge(train_x, train_y, args.alpha)
        cluster_scaler, cluster_model_ridge = fit_ridge(train_aug_x, train_y, args.alpha)

        fold_cluster_metadata.append(
            {
                "fold": int(fold),
                "training_samples": training_names,
                "validation_samples": validation_names,
                "pca_components": int(cluster_model.pca.n_components_),
                "explained_variance_ratio_sum": float(cluster_model.pca.explained_variance_ratio_.sum()),
            }
        )
        for sample in val_samples:
            baseline_pred = predict(baseline_scaler, baseline_model, sample.x)
            cluster_pred = predict(cluster_scaler, cluster_model_ridge, val_aug[sample.sample])
            fold_rows.append(metric_row(sample.sample, f"cv_fold_{fold}", "baseline_ridge", sample.y, baseline_pred))
            fold_rows.append(metric_row(sample.sample, f"cv_fold_{fold}", "cluster_aware_ridge", sample.y, cluster_pred))

    # Fit once on all development samples and evaluate the untouched final set.
    final_cluster_model = fit_cluster_model(
        development,
        genes,
        args.n_clusters,
        args.pca_components,
        args.random_state,
    )
    development_aug = {sample.sample: add_cluster_features(sample, final_cluster_model) for sample in development}
    final_aug = {sample.sample: add_cluster_features(sample, final_cluster_model) for sample in final_test}
    train_x, train_y = concatenate(development)
    train_aug_x, _ = concatenate(development, development_aug)
    final_x, final_y = concatenate(final_test)
    final_aug_x, _ = concatenate(final_test, final_aug)

    baseline_scaler, baseline_model = fit_ridge(train_x, train_y, args.alpha)
    cluster_scaler, cluster_model_ridge = fit_ridge(train_aug_x, train_y, args.alpha)
    final_rows = []
    prediction_rows = []
    for sample in final_test:
        baseline_pred = predict(baseline_scaler, baseline_model, sample.x)
        cluster_pred = predict(cluster_scaler, cluster_model_ridge, final_aug[sample.sample])
        final_rows.append(metric_row(sample.sample, "final_test", "baseline_ridge", sample.y, baseline_pred))
        final_rows.append(metric_row(sample.sample, "final_test", "cluster_aware_ridge", sample.y, cluster_pred))
        for index, (target, baseline, cluster) in enumerate(zip(sample.y, baseline_pred, cluster_pred)):
            prediction_rows.append(
                {
                    "sample": sample.sample,
                    "tile_index": index,
                    "x": float(sample.coords[index, 0]),
                    "y": float(sample.coords[index, 1]),
                    "xenium_egfr": float(target),
                    "baseline_prediction": float(baseline),
                    "cluster_aware_prediction": float(cluster),
                }
            )

    # Pooled rows make the primary comparison easy to read; per-sample rows
    # preserve the sample-level behavior and expose possible heterogeneity.
    for group, rows in [("development_cv", fold_rows), ("final_test", final_rows)]:
        if group == "development_cv":
            for method in ["baseline_ridge", "cluster_aware_ridge"]:
                method_rows = [row for row in rows if row["method"] == method]
                y_parts = []
                p_parts = []
                for row in method_rows:
                    sample = samples[row["sample"]]
                    if group == "development_cv":
                        # Recompute pooled CV predictions below from the saved
                        # per-sample metric inputs is not possible, so pooled
                        # values are intentionally represented by the mean of
                        # fold/sample metrics in the CSV.
                        pass
                if method_rows:
                    rows.append(
                        {
                            "sample": "__mean_of_sample_rows__",
                            "group": group,
                            "method": method,
                            "n_tiles": int(sum(row["n_tiles"] for row in method_rows)),
                            "mean_target": float(np.mean([row["mean_target"] for row in method_rows])),
                            "mean_prediction": float(np.mean([row["mean_prediction"] for row in method_rows])),
                            "rmse": float(np.mean([row["rmse"] for row in method_rows])),
                            "mae": float(np.mean([row["mae"] for row in method_rows])),
                            "pearson_r": float(np.nanmean([row["pearson_r"] for row in method_rows])),
                            "spearman_r": float(np.nanmean([row["spearman_r"] for row in method_rows])),
                        }
                    )

    # Add pooled final-test metrics from the actual predictions.
    baseline_final_pred = []
    cluster_final_pred = []
    for sample in final_test:
        baseline_final_pred.append(predict(baseline_scaler, baseline_model, sample.x))
        cluster_final_pred.append(predict(cluster_scaler, cluster_model_ridge, final_aug[sample.sample]))
    final_rows.append(metric_row("__pooled__", "final_test", "baseline_ridge", final_y, np.concatenate(baseline_final_pred)))
    final_rows.append(metric_row("__pooled__", "final_test", "cluster_aware_ridge", final_y, np.concatenate(cluster_final_pred)))

    pd.DataFrame(fold_rows + final_rows).to_csv(OUT_ROOT / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT_ROOT / "final_test_predictions.csv", index=False)
    pd.DataFrame(fold_cluster_metadata).to_json(OUT_ROOT / "cv_cluster_metadata.json", orient="records", indent=2)

    metadata = {
        "development_samples": development_names,
        "final_test_samples": final_names,
        "n_input_genes": len(genes),
        "egfr_feature_index": genes.index("EGFR"),
        "egfr_excluded_from_cluster_construction": True,
        "n_clusters": args.n_clusters,
        "pca_components": args.pca_components,
        "ridge_alpha": args.alpha,
        "cluster_aware_feature_count": int(train_aug_x.shape[1]),
        "cluster_feature_definition": "current tile one-hot plus eight-nearest-tile cluster fractions",
        "genes": genes,
    }
    (OUT_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    pooled = pd.DataFrame(final_rows).query("sample == '__pooled__'")
    print(pooled[["method", "rmse", "mae", "pearson_r", "spearman_r"]].to_string(index=False))
    print(f"Wrote results to {OUT_ROOT}")


if __name__ == "__main__":
    main()
