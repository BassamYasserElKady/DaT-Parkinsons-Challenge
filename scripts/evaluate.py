#!/usr/bin/env python3
"""Evaluate final-test predictions; this script never trains or selects a model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_pipeline.io import load_processed_sample
from spatial_pipeline.manifest import load_manifest
from spatial_pipeline.metrics import evaluate_per_sample_and_pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-count", type=int, default=8)
    parser.add_argument("--final-test-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    processed_dir = args.processed_dir if args.processed_dir.is_absolute() else root / args.processed_dir
    predictions_dir = args.predictions_dir if args.predictions_dir.is_absolute() else root / args.predictions_dir
    output_path = args.output if args.output.is_absolute() else root / args.output
    records = load_manifest(manifest_path, root, expected_development=args.development_count, expected_final_test=args.final_test_count)
    final_records = [record for record in records if record.split == "final_test"]
    evaluated = []
    for record in final_records:
        sample = load_processed_sample(processed_dir / f"{record.sample_id}.npz")
        prediction = np.load(predictions_dir / f"{record.sample_id}.npy")
        if prediction.shape != sample.targets_log1p_cp10k.shape:
            raise ValueError(f"prediction shape mismatch for {record.sample_id}")
        evaluated.append((record.sample_id, prediction, sample.targets_log1p_cp10k, sample.genes))
    metrics = evaluate_per_sample_and_pooled(evaluated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
