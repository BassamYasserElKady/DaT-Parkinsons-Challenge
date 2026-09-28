#!/usr/bin/env python
"""Marker-score Xenium cells and summarize EGFR by candidate population.

This is intentionally a transparent, panel-aware annotation rather than a
black-box cell-type classifier.  The 10x breast panel is targeted, so absence
of a marker is not treated as proof that a cell type is absent.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.spatial_pipeline.xenium_archive import NestedZipStore


SAMPLE_RE = re.compile(r"Human_Breast_Biomarkers_(?P<sample>S\d+_(?:Bot|Mid|Top))")

# Positive markers are deliberately limited to genes in the Xenium panel.
# Scores are the mean z-scored log expression of available positive markers,
# minus the mean of available exclusion markers.
MARKERS = {
    "epithelial": ["EPCAM", "KRT19", "KRT5", "KRT14", "KRT17", "KRT15", "TACSTD2", "CLDN4", "DSP", "KRT6B"],
    "tumor_epithelial": ["EPCAM", "KRT19", "S100P", "KRT17", "KRT6B", "CEACAM6", "MIA", "FOXC1", "SERPINB5", "TMEM45B", "LYPD3"],
    "proliferation": ["MKI67", "TOP2A", "UBE2C", "BIRC5", "CENPF", "AURKA", "MELK", "NUF2", "TYMS", "RRM2", "CDC20", "NDC80"],
    "luminal": ["ESR1", "PGR", "FOXA1", "GATA3", "AGR3", "SCGB2A1", "SCGB2A2", "TFF3", "PIP", "MLPH", "AR"],
    "her2": ["ERBB2", "GRB7", "ERBB2", "GATA3", "SLC39A6"],
    "basal": ["KRT5", "KRT14", "KRT17", "KRT6B", "FOXC1", "TP63", "COL17A1", "EGFR"],
    "stem_like_candidate": ["ALDH1A3", "KRT17", "BCL2", "KRT5", "EGFR", "MIA"],
    "myoepithelial": ["TP63", "ACTA2", "ACTG2", "COL17A1", "OXTR", "KRT5", "KRT14", "KRT17", "MYLK"],
    "caf_stromal": ["MMP11", "CXCL14", "CXCL12", "SFRP1", "SFRP4", "FBLN1", "C1R", "CCDC80", "IGFBP5", "PDGFD", "PTN", "LOXL1", "TFPI2", "COL4A1"],
    "endothelial": ["PECAM1", "VWF", "PLVAP", "FLT1", "CLEC14A", "AQP1", "STC1", "LAMA4", "CA4"],
    "perivascular": ["RGS5", "PDGFRB", "ACTA2", "MYH11", "TAGLN", "ACTG2", "CNN1", "MYLK", "NDUFA4L2"],
    "myeloid": ["LYZ", "CD68", "CD163", "APOC1", "CHIT1", "MMP12", "MMP9", "C15orf48", "ITGAX", "HMOX1", "S100A8", "S100A9", "SLC40A1"],
    "macrophage": ["CD68", "CD163", "APOC1", "CHIT1", "MMP12", "C15orf48", "HMOX1", "SLC40A1"],
    "inflammatory_myeloid": ["LYZ", "S100A8", "S100A9", "MMP9", "CTSV", "C15orf48", "IL1R1"],
    "t_cell": ["CD3E", "TRAC", "CD8A", "IL7R", "CCL5", "CD27"],
    "b_cell": ["MS4A1", "CD74", "BANK1", "TNFRSF13C"],
    "plasma": ["SDC1", "TENT5C", "MZB1", "CD74"],
    "mast": ["KIT", "CPA3"],
    "adipocyte": ["ADIPOQ", "LPL", "LEPR"],
}

EXCLUDE = {
    "epithelial": ["TRAC", "CD3E", "LYZ", "PECAM1", "VWF", "COL4A1", "PDGFRB"],
    "tumor_epithelial": ["TRAC", "CD3E", "LYZ", "PECAM1", "VWF", "PDGFRB", "COL4A1"],
    "myoepithelial": ["TRAC", "CD3E", "LYZ", "PECAM1", "VWF", "MS4A1"],
    "caf_stromal": ["EPCAM", "KRT19", "TRAC", "CD3E", "LYZ", "PECAM1", "VWF"],
    "endothelial": ["EPCAM", "KRT19", "TRAC", "CD3E", "LYZ", "CD68", "CD163"],
    "perivascular": ["EPCAM", "KRT19", "TRAC", "CD3E", "LYZ", "CD68", "CD163"],
    "myeloid": ["EPCAM", "KRT19", "TRAC", "CD3E", "PECAM1", "VWF"],
    "macrophage": ["EPCAM", "KRT19", "TRAC", "CD3E", "PECAM1", "VWF"],
    "inflammatory_myeloid": ["EPCAM", "KRT19", "TRAC", "CD3E", "PECAM1", "VWF"],
}


def decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def archive_for(folder: Path) -> Path:
    candidates = sorted(folder.glob("*_xe_outs.zip"))
    if not candidates:
        return None
    return candidates[0]


def read_counts(path: Path):
    with zipfile.ZipFile(path) as outer:
        with NestedZipStore(outer, "cell_feature_matrix.zarr.zip") as root:
            f = root["cell_features"]
            names = [decode(v) for v in f.attrs["feature_keys"]]
            types = [decode(v) for v in f.attrs["feature_types"]]
            csc = f["csc"]
            data = np.asarray(csc["data"][:])
            indices = np.asarray(csc["indices"][:])
            indptr = np.asarray(csc["indptr"][:])
            n_cells = len(indptr) - 1
    x = sparse.csc_matrix((data, indices, indptr), shape=(len(names), n_cells)).tocsr()
    gene_idx = [i for i, t in enumerate(types) if t == "gene"]
    gene_names = [names[i] for i in gene_idx]
    x = x[gene_idx]
    return x, gene_names


def score_cells(x: sparse.csr_matrix, genes: list[str]) -> pd.DataFrame:
    gene_to_i = {g: i for i, g in enumerate(genes)}
    lib = np.asarray(x.sum(axis=0)).ravel()
    lib[lib == 0] = 1
    # log1p counts per 10,000 measured gene transcripts.
    norm = x.multiply(10000.0 / lib).tocsr()
    norm.data = np.log1p(norm.data)
    # Marker z-scores are computed across all cells for each measured gene.
    means = np.asarray(norm.mean(axis=1)).ravel()
    sq = np.asarray(norm.multiply(norm).mean(axis=1)).ravel()
    sd = np.sqrt(np.maximum(sq - means * means, 1e-8))

    def gene_z(g):
        if g not in gene_to_i:
            return None
        v = norm.getrow(gene_to_i[g]).toarray().ravel()
        return (v - means[gene_to_i[g]]) / sd[gene_to_i[g]]

    out = pd.DataFrame({"cell_index": np.arange(x.shape[1]), "gene_transcripts": lib})
    out["egfr_count"] = np.asarray(x.getrow(gene_to_i["EGFR"]).toarray()).ravel() if "EGFR" in gene_to_i else 0
    out["egfr_log1p_cp10k"] = norm.getrow(gene_to_i["EGFR"]).toarray().ravel() if "EGFR" in gene_to_i else 0
    available = {}
    for label, positive in MARKERS.items():
        pos = [gene_z(g) for g in positive if gene_z(g) is not None]
        neg = [gene_z(g) for g in EXCLUDE.get(label, []) if gene_z(g) is not None]
        available[label] = (len(pos), len(neg))
        out[f"score_{label}"] = np.mean(pos, axis=0) if pos else 0.0
        if neg:
            out[f"score_{label}"] -= np.mean(neg, axis=0)
    # Transparent broad annotation: select among lineage scores, while using
    # minimum evidence so low-RNA cells are not forced into a type.
    lineage = ["tumor_epithelial", "myoepithelial", "caf_stromal", "endothelial", "perivascular", "myeloid", "t_cell", "b_cell", "plasma", "mast", "adipocyte"]
    vals = out[[f"score_{g}" for g in lineage]].to_numpy()
    ix = vals.argmax(axis=1)
    best = vals[np.arange(len(out)), ix]
    out["predicted_lineage"] = np.where(best >= 0.15, np.array(lineage, dtype=object)[ix], "unresolved")
    # Tumor subtype calls are only made within high epithelial/tumor-score cells.
    tumor = out.score_tumor_epithelial > 0.5
    sub = np.full(len(out), "not_tumor_epithelial", dtype=object)
    sub[tumor] = "other_tumor_epithelial"
    sub[tumor & (out.score_baseline if False else np.zeros(len(out), dtype=bool))] = "basal_like_candidate"
    # Use an explicit combination so this remains valid if marker sets change.
    basal = tumor & (out.score_basal > out.score_luminal) & (out.score_basal > out.score_her2)
    luminal = tumor & (out.score_luminal >= out.score_basal) & (out.score_luminal > out.score_her2)
    her2 = tumor & (out.score_her2 >= out.score_basal) & (out.score_her2 > out.score_luminal)
    sub[basal] = "basal_like_candidate"
    sub[luminal] = "luminal_candidate"
    sub[her2] = "HER2_enriched_candidate"
    out["tumor_subtype_candidate"] = sub
    out["stem_like_candidate"] = tumor & (out.score_stem_like_candidate > 0.5)
    out["proliferative_tumor_candidate"] = tumor & (out.score_proliferation > 0.5)
    return out


def summarize(cells: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []
    for group, g in cells.groupby("predicted_lineage", sort=False):
        rows.append({
            "sample": sample,
            "category": group,
            "cells": len(g),
            "fraction_of_sample": len(g) / len(cells),
            "egfr_positive_cells": int((g.egfr_count > 0).sum()),
            "egfr_detection_fraction": float((g.egfr_count > 0).mean()),
            "egfr_mean_log1p_cp10k": float(g.egfr_log1p_cp10k.mean()),
            "egfr_median_log1p_cp10k": float(g.egfr_log1p_cp10k.median()),
            "egfr_mean_count": float(g.egfr_count.mean()),
        })
    for group, g in cells[cells.tumor_subtype_candidate != "not_tumor_epithelial"].groupby("tumor_subtype_candidate", sort=False):
        rows.append({
            "sample": sample,
            "category": group,
            "cells": len(g),
            "fraction_of_sample": len(g) / len(cells),
            "egfr_positive_cells": int((g.egfr_count > 0).sum()),
            "egfr_detection_fraction": float((g.egfr_count > 0).mean()),
            "egfr_mean_log1p_cp10k": float(g.egfr_log1p_cp10k.mean()),
            "egfr_median_log1p_cp10k": float(g.egfr_log1p_cp10k.median()),
            "egfr_mean_count": float(g.egfr_count.mean()),
        })
    special = {
        "stem_like_candidate": cells[cells.stem_like_candidate],
        "proliferative_tumor_candidate": cells[cells.proliferative_tumor_candidate],
    }
    for group, g in special.items():
        rows.append({
            "sample": sample,
            "category": group,
            "cells": len(g),
            "fraction_of_sample": len(g) / len(cells),
            "egfr_positive_cells": int((g.egfr_count > 0).sum()),
            "egfr_detection_fraction": float((g.egfr_count > 0).mean()) if len(g) else np.nan,
            "egfr_mean_log1p_cp10k": float(g.egfr_log1p_cp10k.mean()) if len(g) else np.nan,
            "egfr_median_log1p_cp10k": float(g.egfr_log1p_cp10k.median()) if len(g) else np.nan,
            "egfr_mean_count": float(g.egfr_count.mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/10x"))
    ap.add_argument("--out", type=Path, default=Path("results/xenium_egfr_cell_annotation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_cells, all_summaries = [], []
    for folder in sorted(args.raw_root.glob("section_*")):
        archive = archive_for(folder)
        if archive is None:
            continue
        match = SAMPLE_RE.search(archive.name)
        if not match:
            continue
        sample = match.group("sample")
        print(f"Reading {sample} ...", flush=True)
        x, genes = read_counts(archive)
        cells = score_cells(x, genes)
        cells.insert(0, "sample", sample)
        cells.to_parquet(args.out / f"{sample}_cell_scores.parquet", index=False)
        all_summaries.append(summarize(cells, sample))
        all_cells.append(cells.groupby("predicted_lineage").agg(cells=("cell_index", "size"), egfr_detection_fraction=("egfr_count", lambda s: float((s > 0).mean())), egfr_mean_count=("egfr_count", "mean"), egfr_mean_log1p_cp10k=("egfr_log1p_cp10k", "mean")).reset_index().assign(sample=sample))
    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(args.out / "egfr_by_candidate_population.csv", index=False)
    broad = pd.concat(all_cells, ignore_index=True)
    broad.to_csv(args.out / "egfr_by_broad_lineage.csv", index=False)
    report = args.out / "README.md"
    report.write_text("""# Xenium EGFR cell annotation\n\n"
This analysis uses the measured Xenium panel to compute transparent marker scores
per cell. `egfr_by_candidate_population.csv` reports EGFR detection and
normalized expression by predicted lineage. Per-cell scores are in the parquet
files. These are candidate labels, not pathology-confirmed cell identities.

The panel lacks several canonical markers (for example COL1A1/DCN/LUM, CD44,
PROM1, and ERBB3), so CAF and cancer-stem-cell calls are especially cautious.
""")


if __name__ == "__main__":
    main()
