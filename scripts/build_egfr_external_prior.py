#!/usr/bin/env python3
"""Build a population-level EGFR co-expression prior from local external data.

This script never uses external samples as patient-specific model inputs. It
only estimates gene-wise association with EGFR and restricts the result to the
project's fitted 276-gene panel.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]


def read_dciss(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="gzip")
    frame["gene_name"] = frame["gene_name"].astype(str).str.upper().str.strip()
    frame = frame.drop_duplicates("gene_name").set_index("gene_name")
    return frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def read_bulk_tar_gz(path: Path) -> pd.DataFrame:
    with tarfile.open(path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError(f"Expected one table in {path}, found {len(members)}")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError(f"Could not read {members[0].name}")
        frame = pd.read_csv(handle, sep="\t")
    frame["Genes"] = frame["Genes"].astype(str).str.upper().str.strip()
    frame = frame.drop_duplicates("Genes").set_index("Genes")
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def normalize_for_correlation(frame: pd.DataFrame, bulk_counts: bool) -> pd.DataFrame:
    values = frame.to_numpy(np.float64)
    if bulk_counts:
        totals = values.sum(axis=0, keepdims=True)
        values = np.log1p(1_000_000.0 * values / np.maximum(totals, 1.0))
    else:
        values = np.log1p(np.maximum(values, 0.0))
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def per_dataset_correlations(frame: pd.DataFrame, genes: list[str], dataset: str, bulk_counts: bool) -> pd.DataFrame:
    normalized = normalize_for_correlation(frame, bulk_counts)
    if "EGFR" not in normalized.index:
        raise ValueError(f"EGFR is absent from {dataset}")
    rows = []
    egfr = normalized.loc["EGFR"].to_numpy()
    for gene in genes:
        if gene not in normalized.index:
            continue
        values = normalized.loc[gene].to_numpy()
        if np.std(values) == 0 or np.std(egfr) == 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "gene": gene,
                "n_samples": len(egfr),
                "pearson_with_egfr": float(pearsonr(values, egfr).statistic),
                "spearman_with_egfr": float(spearmanr(values, egfr).statistic),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--dcis", type=Path, default=Path("data/external/geo/GSE281303/GSE281303_DCISexpData.txt.gz"))
    parser.add_argument("--bulk", type=Path, default=Path("data/external/geo/GSE176078/GSE176078_Wu_etal_2021_bulkRNAseq_raw_counts.txt.gz"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_priors/egfr"))
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = torch.load(args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint, map_location="cpu", weights_only=False)
    genes = [str(g).upper() for g in checkpoint["genes"]]
    dcis = read_dciss(args.dcis if args.dcis.is_absolute() else root / args.dcis)
    bulk = read_bulk_tar_gz(args.bulk if args.bulk.is_absolute() else root / args.bulk)
    dcis_corr = per_dataset_correlations(dcis, genes, "GSE281303_DCIS", bulk_counts=False)
    bulk_corr = per_dataset_correlations(bulk, genes, "GSE176078_bulkRNAseq", bulk_counts=True)
    correlations = pd.concat([dcis_corr, bulk_corr], ignore_index=True)
    grouped = correlations.groupby("gene", as_index=False).agg(
        datasets=("dataset", "nunique"),
        total_samples=("n_samples", "sum"),
        mean_pearson=("pearson_with_egfr", "mean"),
        mean_spearman=("spearman_with_egfr", "mean"),
        min_pearson=("pearson_with_egfr", "min"),
        max_pearson=("pearson_with_egfr", "max"),
    )
    grouped["stable_positive"] = (grouped["datasets"] == 2) & (grouped["min_pearson"] > 0)
    grouped["stable_negative"] = (grouped["datasets"] == 2) & (grouped["max_pearson"] < 0)
    positive = grouped[grouped["stable_positive"]].sort_values(["mean_pearson", "mean_spearman"], ascending=False).head(args.top_n)
    negative = grouped[grouped["stable_negative"]].sort_values(["mean_pearson", "mean_spearman"], ascending=True).head(args.top_n)
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(output / "dataset_gene_correlations.csv", index=False)
    grouped.to_csv(output / "egfr_panel_prior.csv", index=False)
    positive[["gene", "datasets", "total_samples", "mean_pearson", "mean_spearman", "min_pearson", "max_pearson"]].to_csv(output / "positive_module_genes.csv", index=False)
    negative[["gene", "datasets", "total_samples", "mean_pearson", "mean_spearman", "min_pearson", "max_pearson"]].to_csv(output / "negative_module_genes.csv", index=False)
    (output / "positive_module_genes.txt").write_text("\n".join(positive["gene"].tolist()) + "\n")
    report = [
        "# External EGFR population prior",
        "",
        "Sources used: GSE281303 DCIS expression and GSE176078 breast bulk RNA-seq. External samples were used only to estimate population-level gene associations; they are not patient-specific conditioning inputs.",
        "",
        f"- Fitted project panel genes: `{len(genes)}`",
        f"- GSE281303 samples: `{dcis.shape[1]}`",
        f"- GSE176078 samples: `{bulk.shape[1]}`",
        f"- Positive module genes retained: `{len(positive)}`",
        "",
        "## Positive EGFR-associated module",
        "",
        ", ".join(positive["gene"].tolist()),
        "",
        "The prior is suitable for regularization or auxiliary multi-task losses. It does not provide spatial labels and must be validated using development Xenium samples only.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n")
    print(f"output={output}")
    print(f"dcis_shape={dcis.shape}; bulk_shape={bulk.shape}; panel_genes={len(genes)}")
    print("positive_module=")
    print(positive[["gene", "mean_pearson", "mean_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
