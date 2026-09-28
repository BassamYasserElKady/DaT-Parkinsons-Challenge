#!/usr/bin/env python3
"""GrowCut-style competitive segmentation on ANTs-registered DaT scans."""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

import validate_atlas_free_striatal_segmentation as pipeline


BACKGROUND_LABEL = 5
NEIGHBOUR_OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def growcut(data: np.ndarray, allowed: np.ndarray, seeds: dict[str, tuple[int, int, int]], intensity_scale: float) -> np.ndarray:
    labels = np.zeros(data.shape, dtype=np.uint8)
    strengths = np.zeros(data.shape, dtype=np.float32)
    queue: list[tuple[float, tuple[int, int, int]]] = []
    for name, point in seeds.items():
        label = pipeline.ROI_LABELS[name]
        labels[point] = label
        strengths[point] = 1.0
        heapq.heappush(queue, (-1.0, point))

    # The exterior of the central allowed region acts as Slicer's background seed.
    for point in zip(*np.where(allowed)):
        x, y, z = point
        if any(
            not (0 <= x + dx < data.shape[0] and 0 <= y + dy < data.shape[1] and 0 <= z + dz < data.shape[2])
            or not allowed[x + dx, y + dy, z + dz]
            for dx, dy, dz in NEIGHBOUR_OFFSETS
        ):
            if labels[point] == 0:
                labels[point] = BACKGROUND_LABEL
                strengths[point] = 1.0
                heapq.heappush(queue, (-1.0, point))

    while queue:
        neg_strength, point = heapq.heappop(queue)
        strength = -neg_strength
        if strength + 1e-7 < float(strengths[point]):
            continue
        source_label = int(labels[point])
        x, y, z = point
        source_value = float(data[point])
        for dx, dy, dz in NEIGHBOUR_OFFSETS:
            neighbour = (x + dx, y + dy, z + dz)
            nx, ny, nz = neighbour
            if not (0 <= nx < data.shape[0] and 0 <= ny < data.shape[1] and 0 <= nz < data.shape[2]) or not allowed[neighbour]:
                continue
            similarity = max(0.0, 1.0 - abs(source_value - float(data[neighbour])) / intensity_scale)
            candidate_strength = strength * similarity
            if candidate_strength > float(strengths[neighbour]) + 1e-7:
                strengths[neighbour] = candidate_strength
                labels[neighbour] = source_label
                heapq.heappush(queue, (-candidate_strength, neighbour))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    labels = pipeline.load_labels(args.labels)
    paths = sorted(path for path in args.registered_dir.glob("*.nii.gz") if path.stem.removesuffix(".nii") in labels)
    y = np.asarray([labels[path.stem.removesuffix(".nii")] for path in paths])
    if args.sample_size == len(paths):
        selected_paths = paths
    else:
        _, selected = train_test_split(np.arange(len(paths)), test_size=args.sample_size, stratify=y, random_state=args.seed)
        selected_paths = [paths[index] for index in sorted(selected)]
    figure_dir, mask_dir = args.output_dir / "figures", args.output_dir / "masks"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, path in enumerate(selected_paths, start=1):
        uid = path.stem.removesuffix(".nii")
        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        data = np.where(np.isfinite(data), np.maximum(data, 0), 0)
        brain = pipeline.brain_mask(data)
        bounds = pipeline.bounding_box(brain)
        midline = pipeline.find_midline(data, brain, bounds)
        allowed = pipeline.central_seed_zone(brain, bounds, midline, "left") | pipeline.central_seed_zone(brain, bounds, midline, "right")
        seeds = {}
        for side in ("left", "right"):
            zone = pipeline.central_seed_zone(brain, bounds, midline, side)
            caudate, putamen = pipeline.select_seed_pair(data, zone, side, np.asarray(image.header.get_zooms()[:3]), 12.0, (2,))
            seeds[f"{side}_caudate"] = caudate
            seeds[f"{side}_putamen"] = putamen
        values = data[allowed]
        scale = max(float(np.percentile(values, 99.5) - np.percentile(values, 1.0)), 1e-6)
        grown = growcut(data, allowed, seeds, scale)
        masks = {name: grown == label for name, label in pipeline.ROI_LABELS.items()}
        encoded = grown.copy()
        encoded[~allowed] = 0
        mask_path = mask_dir / f"{uid}_growcut_rois.nii.gz"
        figure_path = figure_dir / f"{uid}_growcut_rois.png"
        nib.save(nib.Nifti1Image(encoded, image.affine, image.header), str(mask_path))
        pipeline.save_figure(uid, data, masks, seeds, figure_path)
        rows.append({
            "uid": uid, "is_pathologic": labels[uid], "midline_x": midline,
            "left_caudate_voxels": int(masks["left_caudate"].sum()),
            "left_putamen_voxels": int(masks["left_putamen"].sum()),
            "right_caudate_voxels": int(masks["right_caudate"].sum()),
            "right_putamen_voxels": int(masks["right_putamen"].sum()),
            "background_voxels": int((grown == BACKGROUND_LABEL).sum()),
            "mask_file": str(mask_path), "figure_file": str(figure_path),
        })
        print(f"[{number}/{len(selected_paths)}] {uid}", flush=True)
    with (args.output_dir / "growcut_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "README.md").write_text(
        "# GrowCut-style registered striatal validation\n\n"
        "Four registered-space striatal seed labels competed against a background label within a central brain search region. "
        "Voxel strength propagated through 26-neighbour intensity similarity. This is a local GrowCut-style implementation; 3D Slicer was not installed in the environment.\n",
        encoding="utf-8",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
