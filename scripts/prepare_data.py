#!/usr/bin/env python3
"""Prepare per-sample matrices; this script never trains a model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_pipeline.io import save_normalization_stats, save_processed_sample
from spatial_pipeline.manifest import load_manifest
from spatial_pipeline.normalization import fit_training_stats
from spatial_pipeline.preprocess import (
    common_development_genes,
    load_and_process_sample,
    restrict_to_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-qv", type=int, default=20)
    parser.add_argument("--development-count", type=int, default=8)
    parser.add_argument("--final-test-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output if args.output.is_absolute() else root / args.output
    records = load_manifest(manifest_path, root, expected_development=args.development_count, expected_final_test=args.final_test_count)

    # This is the only gene-vocabulary derivation step. It uses development
    # samples only; final-test genes are checked against the resulting list.
    processed = [load_and_process_sample(record, min_qv=args.min_qv) for record in records]
    genes = common_development_genes(processed)
    aligned = [restrict_to_genes(sample, genes) for sample in processed]
    development = [sample for sample in aligned if sample.split == "development"]
    stats = fit_training_stats(development)

    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in aligned:
        save_processed_sample(sample, output_dir)
    save_normalization_stats(stats, genes, output_dir / "development_normalization.npz")
    np.savetxt(output_dir / "development_genes.txt", np.asarray(genes, dtype=str), fmt="%s")
    print(f"Prepared {len(aligned)} samples with {len(genes)} development-defined genes")
    print(f"Wrote per-sample matrices and development-only statistics to {output_dir}")


if __name__ == "__main__":
    main()
