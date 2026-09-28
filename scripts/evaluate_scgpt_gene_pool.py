#!/usr/bin/env python3
"""Test whether local scGPT gene similarity predicts EGFR spatial hints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scgpt-dir", type=Path, default=Path("models/scgpt_whole_human"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/adapter_baseline/adapter_final.pt"))
    parser.add_argument("--hint-metrics", type=Path, default=Path("results/egfr_tile_hints/gene_egfr_correlations_by_sample.csv"))
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scgpt_egfr_gene_pool"))
    args = parser.parse_args()
    root = args.root.resolve()
    scgpt_dir = args.scgpt_dir if args.scgpt_dir.is_absolute() else root / args.scgpt_dir
    checkpoint = torch.load(args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint, map_location="cpu", weights_only=False)
    genes = [str(g).upper() for g in checkpoint["genes"]]
    vocab = json.loads((scgpt_dir / "vocab.json").read_text())
    weights = torch.load(scgpt_dir / "best_model.pt", map_location="cpu", weights_only=True)
    embedding = weights["encoder.embedding.weight"].float().numpy()
    egfr_token = vocab.get("EGFR")
    if egfr_token is None:
        raise ValueError("EGFR is absent from the local scGPT vocabulary")
    egfr_vector = embedding[int(egfr_token)]
    cosine = embedding @ egfr_vector / (np.linalg.norm(embedding, axis=1) * max(np.linalg.norm(egfr_vector), 1e-12))

    hint_path = args.hint_metrics if args.hint_metrics.is_absolute() else root / args.hint_metrics
    by_sample = pd.read_csv(hint_path)
    dev = by_sample[by_sample["split"] == "development"]
    dev_summary = dev.groupby("gene", as_index=False).agg(
        development_mean_pearson=("spatial_pearson_with_egfr", "mean"),
        development_mean_spearman=("spatial_spearman_with_egfr", "mean"),
        development_positive_fraction=("spatial_spearman_with_egfr", lambda x: float(np.mean(x > 0))),
    )
    test = by_sample[by_sample["split"] == "final_test"]
    test_summary = test.groupby("gene", as_index=False).agg(
        final_test_mean_pearson=("spatial_pearson_with_egfr", "mean"),
        final_test_mean_spearman=("spatial_spearman_with_egfr", "mean"),
    )

    rows = []
    for gene in genes:
        token = vocab.get(gene)
        if token is None or gene == "EGFR":
            continue
        row = dev_summary.loc[dev_summary["gene"] == gene]
        if len(row) != 1:
            continue
        test_row = test_summary.loc[test_summary["gene"] == gene]
        record = {
            "gene": gene,
            "scgpt_vocab_token": int(token),
            "scgpt_cosine_to_egfr": float(cosine[int(token)]),
            **row.iloc[0].to_dict(),
        }
        if len(test_row) == 1:
            record.update(test_row.iloc[0].to_dict())
        rows.append(record)
    table = pd.DataFrame(rows).sort_values("scgpt_cosine_to_egfr", ascending=False).reset_index(drop=True)
    table["scgpt_rank"] = np.arange(1, len(table) + 1)

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "scgpt_gene_similarity_and_spatial_metrics.csv", index=False)

    summary_rows = []
    for k in (10, 25, 50, 100):
        selected = table.head(min(k, len(table)))
        summary_rows.append(
            {
                "pool": f"scgpt_top_{k}",
                "n_genes": len(selected),
                "mean_development_pearson": selected["development_mean_pearson"].mean(),
                "mean_development_spearman": selected["development_mean_spearman"].mean(),
                "fraction_development_spearman_positive": (selected["development_mean_spearman"] > 0).mean(),
                "mean_final_test_pearson": selected["final_test_mean_pearson"].mean(),
                "mean_final_test_spearman": selected["final_test_mean_spearman"].mean(),
            }
        )
    pool_summary = pd.DataFrame(summary_rows)
    pool_summary.to_csv(output / "scgpt_pool_metrics.csv", index=False)

    dev_cosine_pearson = float(spearmanr(table["scgpt_cosine_to_egfr"], table["development_mean_pearson"]).statistic)
    dev_cosine_spearman = float(spearmanr(table["scgpt_cosine_to_egfr"], table["development_mean_spearman"]).statistic)
    top_genes = table.head(args.top_n)["gene"].tolist()
    report = [
        "# scGPT EGFR gene-pool evaluation",
        "",
        "scGPT cosine similarity to EGFR was computed from the local frozen gene-embedding checkpoint. Xenium spatial correlations were used only to evaluate whether that similarity predicts a useful EGFR-associated gene pool.",
        "",
        f"- Project genes evaluated: `{len(table)}`",
        f"- scGPT-to-development spatial Pearson rank correlation: `{dev_cosine_pearson:.4f}`",
        f"- scGPT-to-development spatial Spearman rank correlation: `{dev_cosine_spearman:.4f}`",
        "",
        "## scGPT top-pool performance",
        "",
        pool_summary.to_markdown(index=False),
        "",
        f"## Top {args.top_n} scGPT genes",
        "",
        ", ".join(top_genes),
        "",
        "The final-test columns are descriptive only; the scGPT pool was selected without final-test Xenium values.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    print(f"output={output}")
    print(f"scgpt_rank_vs_dev_pearson={dev_cosine_pearson:.6f}")
    print(f"scgpt_rank_vs_dev_spearman={dev_cosine_spearman:.6f}")
    print(pool_summary.to_string(index=False))
    print("top_genes=" + ",".join(top_genes))


if __name__ == "__main__":
    main()
