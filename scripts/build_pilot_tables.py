#!/usr/bin/env python3
"""Create one local DeepSpot-M/Xenium tile-table pilot; never trains.

This exporter is deliberately limited to one selected sample per invocation.
It uses the local published DeepSpot-M checkpoint and the cell expression plus
cell-centroid arrays already present inside the nested Xenium ZIP archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import tifffile
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_pipeline.xenium_archive import read_cell_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_manifest.csv"))
    parser.add_argument("--sample-id", default="S1_Top")
    parser.add_argument("--source", default="scgpt")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deepspotm"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot_s1_top_scgpt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parse_affine(value: str) -> np.ndarray:
    values = [float(part.strip()) for part in str(value).split(";")]
    if len(values) != 9:
        raise ValueError("alignment_affine_row_major must contain 9 values")
    affine = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine)) < 1e-12:
        raise ValueError("alignment affine must be finite and invertible")
    return affine


def _tile_cells(row: pd.Series, centroids_um: np.ndarray, tile_size: int = 224):
    """Map Xenium cell centroids to full-resolution H&E tile coordinates."""

    xenium_px = centroids_um / float(row["xenium_pixel_size_um"])
    points = np.column_stack([xenium_px, np.ones(len(xenium_px))])
    small = (np.linalg.inv(_parse_affine(row["alignment_affine_row_major"])) @ points.T).T
    small_x = small[:, 0] / small[:, 2]
    small_y = small[:, 1] / small[:, 2]
    # The current 10x alignment CSV maps full-resolution H&E pixels directly
    # to Xenium pixels. The manifest he_tile_* fields describe TIFF storage
    # chunks (1024 px), not a registration-scale image.
    full_x = small_x
    full_y = small_y
    inside = (
        (full_x >= 0)
        & (full_x < float(row["he_width_px"]))
        & (full_y >= 0)
        & (full_y < float(row["he_height_px"]))
    )
    tile_x = np.floor(full_x / tile_size).astype(np.int64) * tile_size
    tile_y = np.floor(full_y / tile_size).astype(np.int64) * tile_size
    return tile_x, tile_y, inside


def _aggregate_targets(row: pd.Series, arrays, model_genes: list[str]):
    from scipy.sparse import csc_matrix, csr_matrix

    names = np.asarray(arrays.feature_names, dtype=object)
    types = np.asarray(arrays.feature_types, dtype=object)
    gene_mask = types == "gene"
    name_to_index = {str(name): int(i) for i, name in enumerate(names) if gene_mask[i]}
    shared = [gene for gene in model_genes if gene in name_to_index]
    if not shared:
        raise ValueError("no shared DeepSpot-M/Xenium genes")
    feature_indices = np.asarray([name_to_index[gene] for gene in shared], dtype=np.int64)

    tile_x, tile_y, inside = _tile_cells(row, arrays.centroids_um)
    # Edge tiles do not form complete 224x224 DeepSpot-M inputs and are
    # omitted, matching the full-tile convention of the legacy table.
    complete = (
        inside
        & (tile_x + 224 <= int(row["he_width_px"]))
        & (tile_y + 224 <= int(row["he_height_px"]))
    )
    cell_indices = np.flatnonzero(complete)
    cell_pairs = pd.MultiIndex.from_arrays([tile_x[cell_indices], tile_y[cell_indices]])
    inverse, unique_pairs = pd.factorize(cell_pairs, sort=True)
    tile_x_unique = unique_pairs.get_level_values(0).to_numpy(dtype=np.int64)
    tile_y_unique = unique_pairs.get_level_values(1).to_numpy(dtype=np.int64)
    n_tiles = len(unique_pairs)

    matrix = csc_matrix(
        (arrays.data, arrays.indices, arrays.indptr),
        shape=(arrays.n_features, arrays.n_cells),
    )
    selected = matrix[feature_indices][:, cell_indices]
    cell_to_tile = csr_matrix(
        (
            np.ones(len(cell_indices), dtype=np.float32),
            (np.arange(len(cell_indices)), inverse),
        ),
        shape=(len(cell_indices), n_tiles),
    )
    counts = (selected @ cell_to_tile).toarray().T.astype(np.float32, copy=False)
    totals = counts.sum(axis=1, keepdims=True)
    targets = np.log1p(10000.0 * counts / np.clip(totals, 1e-8, None)).astype(np.float32)
    return shared, tile_x_unique, tile_y_unique, targets, len(cell_indices), int(inside.sum())


def _load_image_array(path: Path):
    store = tifffile.imread(path, aszarr=True)
    root = zarr.open(store, mode="r")
    return store, root["0"]


def _predict_tiles(model, transform, image, tile_x, tile_y, genes, device, batch_size, amp):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(tile_x), batch_size):
            batch = []
            for x, y in zip(tile_x[start : start + batch_size], tile_y[start : start + batch_size]):
                patch = np.asarray(image[int(y) : int(y) + 224, int(x) : int(x) + 224, :])
                batch.append(transform(patch))
            pixels = torch.stack(batch).to(device, non_blocking=True)
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model.predict_genes(pixels, genes)
            else:
                output = model.predict_genes(pixels, genes)
            outputs.append(output.float().cpu().numpy())
            print(f"Predicted {min(start + batch_size, len(tile_x)):,}/{len(tile_x):,} tiles", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    root = args.root.resolve()
    manifest_path = _resolve(root, args.manifest)
    frame = pd.read_csv(manifest_path)
    matches = frame.loc[frame["sample_id"].astype(str) == args.sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest row for {args.sample_id!r}")
    row = matches.iloc[0]

    archive_path = _resolve(root, row["xenium_archive_path"])
    image_path = _resolve(root, row["he_image_path"])
    arrays = read_cell_arrays(archive_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from deepspotm import DeepSpotM

    model, transform = DeepSpotM.from_pretrained(
        str(_resolve(root, args.model_dir)), source=args.source, device=device,
    )
    genes, tile_x, tile_y, targets, cells_in_complete_tiles, cells_inside = _aggregate_targets(
        row, arrays, list(model.gene_names),
    )
    if args.max_tiles is not None:
        if args.max_tiles <= 0:
            raise ValueError("--max-tiles must be positive")
        if len(tile_x) > args.max_tiles:
            selected = np.linspace(0, len(tile_x) - 1, args.max_tiles, dtype=np.int64)
            tile_x, tile_y, targets = tile_x[selected], tile_y[selected], targets[selected]

    store, image = _load_image_array(image_path)
    try:
        predictions = _predict_tiles(
            model, transform, image, tile_x, tile_y, genes, device, args.batch_size, args.amp,
        )
    finally:
        store.close()

    spots = np.asarray([f"{args.sample_id}_{x}_{y}" for x, y in zip(tile_x, tile_y)])
    prediction_frame = pd.DataFrame(predictions, columns=genes)
    prediction_frame.insert(0, "y", tile_y)
    prediction_frame.insert(0, "x", tile_x)
    prediction_frame.insert(0, "spot", spots)
    target_frame = pd.DataFrame(targets, columns=genes)
    target_frame.insert(0, "y", tile_y)
    target_frame.insert(0, "x", tile_x)
    target_frame.insert(0, "spot", spots)

    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{args.sample_id.lower()}_deepspot_{args.source}_predictions.parquet"
    target_path = output_dir / f"{args.sample_id.lower()}_xenium_log1p_cp10k_targets.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    target_frame.to_parquet(target_path, index=False)
    metadata = {
        "sample_id": args.sample_id,
        "source": args.source,
        "device": str(device),
        "model_dir": str(_resolve(root, args.model_dir)),
        "image_path": str(image_path),
        "archive_path": str(archive_path),
        "n_cells": arrays.n_cells,
        "cells_inside_full_he": cells_inside,
        "cells_in_complete_tiles": cells_in_complete_tiles,
        "n_tiles": len(tile_x),
        "max_tiles": args.max_tiles,
        "amp": args.amp,
        "n_shared_genes": len(genes),
        "genes": genes,
        "target_definition": "log1p(CP10K) from Xenium cell feature counts aggregated by cell centroid tile",
        "tile_size_px": 224,
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("sample_id", "source", "device", "n_cells", "n_tiles", "n_shared_genes")}, indent=2))


if __name__ == "__main__":
    main()
