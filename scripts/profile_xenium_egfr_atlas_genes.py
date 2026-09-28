#!/usr/bin/env python3
"""Create gene-level, pseudobulk profiles for the joint Xenium cell states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.annotate_xenium_egfr_cells import read_counts
from scripts.build_xenium_egfr_cell_atlas import fit_states, sample_annotation, state_names, assign_cells


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sample_manifest.csv"
ANNOTATION_ROOT = ROOT / "results" / "xenium_egfr_cell_annotation"
OUT = ROOT / "results" / "xenium_egfr_cell_atlas"


def main() -> None:
    manifest = pd.read_csv(MANIFEST).rename(columns={"sample_id": "sample"})
    samples = manifest["sample"].astype(str).tolist()
    annotations = {s: sample_annotation(s) for s in samples}
    scaler, model = fit_states(annotations, 12, 6000, 17)
    names = state_names(scaler, model)
    assigned = {s: assign_cells(frame, scaler, model, names) for s, frame in annotations.items()}

    pseudobulk = []
    for row in manifest.itertuples(index=False):
        sample = row.sample
        archive = ROOT / row.xenium_archive_path
        print(f"Profiling {sample} ...", flush=True)
        matrix, genes = read_counts(archive)
        cells = assigned[sample]
        states = cells.sort_values("cell_index")["state_id"].to_numpy()
        if len(states) != matrix.shape[1]:
            raise ValueError(f"cell alignment mismatch for {sample}")
        totals = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        for state_id in range(12):
            in_state = states == state_id
            rest = ~in_state
            for label, mask in (("state", in_state), ("rest", rest)):
                if not mask.any():
                    continue
                counts = np.asarray(matrix[:, mask].sum(axis=1)).ravel().astype(float)
                library = counts.sum()
                cp10k = 10000.0 * counts / max(library, 1.0)
                pseudobulk.extend(
                    {
                        "sample": sample,
                        "state_id": state_id,
                        "state": names[state_id],
                        "contrast_group": label,
                        "n_cells": int(mask.sum()),
                        "gene": gene,
                        "mean_log1p_cp10k_pseudobulk": float(np.log1p(cp10k[i])),
                        "detected_fraction": float((matrix[i, mask] > 0).mean()),
                    }
                    for i, gene in enumerate(genes)
                )

    long = pd.DataFrame(pseudobulk)
    long.to_csv(OUT / "state_gene_profiles_by_sample.csv", index=False)
    state = long[long.contrast_group.eq("state")].copy()
    rest = long[long.contrast_group.eq("rest")].copy()
    keys = ["sample", "state_id", "state", "gene"]
    joined = state.merge(rest, on=keys, suffixes=("_state", "_rest"))
    joined["delta_log1p_cp10k"] = joined["mean_log1p_cp10k_pseudobulk_state"] - joined["mean_log1p_cp10k_pseudobulk_rest"]
    joined["log2fc_pseudobulk"] = np.log2((np.expm1(joined["mean_log1p_cp10k_pseudobulk_state"]) + 1e-3) / (np.expm1(joined["mean_log1p_cp10k_pseudobulk_rest"]) + 1e-3))
    joined.to_csv(OUT / "state_gene_effects_by_sample.csv", index=False)

    pooled = (joined.groupby(["state_id", "state", "gene"], as_index=False)
              .agg(samples=("sample", "nunique"), mean_log2fc=("log2fc_pseudobulk", "mean"),
                   median_log2fc=("log2fc_pseudobulk", "median"), mean_delta_log1p=("delta_log1p_cp10k", "mean")))
    pooled["abs_mean_log2fc"] = pooled.mean_log2fc.abs()
    pooled = pooled.sort_values(["state_id", "mean_log2fc"], ascending=[True, False])
    pooled.to_csv(OUT / "state_gene_effects_pooled.csv", index=False)
    top = pd.concat([g.head(25).assign(direction="up") for _, g in pooled.groupby(["state_id", "state"], sort=True)], ignore_index=True)
    down = pd.concat([g.tail(25).sort_values("mean_log2fc").assign(direction="down") for _, g in pooled.groupby(["state_id", "state"], sort=True)], ignore_index=True)
    pd.concat([top, down], ignore_index=True).to_csv(OUT / "state_top_gene_profiles.csv", index=False)
    print(json.dumps({"samples": len(samples), "genes": len(long.gene.unique()), "rows": len(long), "states": names}, indent=2))


if __name__ == "__main__":
    main()
