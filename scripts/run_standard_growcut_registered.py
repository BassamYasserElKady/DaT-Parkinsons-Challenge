#!/usr/bin/env python3
"""Run a plain full-volume GrowCut-style competition from four seed labels."""

from __future__ import annotations

import argparse
import csv
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

import validate_atlas_free_striatal_segmentation as pipeline


NEIGHBOURS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def growcut(data: np.ndarray, seeds: dict[str, tuple[int, int, int]], scale: float) -> np.ndarray:
    """Competitive GrowCut propagation over the complete image volume.

    There is no search mask, background seed, region threshold, radius limit,
    or post-processing. Each seed starts with strength 1 and competes by local
    intensity similarity, matching the core Grow from Seeds behavior.
    """
    labels = np.zeros(data.shape, dtype=np.uint8)
    strengths = np.zeros(data.shape, dtype=np.float32)
    queue: list[tuple[float, tuple[int, int, int]]] = []
    for name, point in seeds.items():
        label = pipeline.ROI_LABELS[name]
        labels[point] = label
        strengths[point] = 1.0
        heapq.heappush(queue, (-1.0, point))

    scale = max(float(scale), 1e-6)
    shape = data.shape
    while queue:
        neg_strength, point = heapq.heappop(queue)
        strength = -neg_strength
        if strength + 1e-7 < float(strengths[point]):
            continue
        source_label = int(labels[point])
        x, y, z = point
        source_value = float(data[point])
        for dx, dy, dz in NEIGHBOURS:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue
            similarity = max(0.0, 1.0 - abs(source_value - float(data[nx, ny, nz])) / scale)
            candidate = strength * similarity
            if candidate > float(strengths[nx, ny, nz]) + 1e-7:
                strengths[nx, ny, nz] = candidate
                labels[nx, ny, nz] = source_label
                heapq.heappush(queue, (-candidate, (nx, ny, nz)))
    return labels


def process_one(payload: tuple[str, int, str]) -> dict[str, object]:
    path_string, label, output_string = payload
    path, output = Path(path_string), Path(output_string)
    uid = path.stem.removesuffix(".nii")
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0)
    brain = pipeline.brain_mask(data)
    bounds = pipeline.bounding_box(brain)
    midline = pipeline.find_midline(data, brain, bounds)
    seeds: dict[str, tuple[int, int, int]] = {}
    spacing = np.asarray(image.header.get_zooms()[:3])
    for side in ("left", "right"):
        zone = pipeline.central_seed_zone(brain, bounds, midline, side)
        caudate, putamen = pipeline.select_seed_pair(data, zone, side, spacing, 12.0, (2,))
        seeds[f"{side}_caudate"] = caudate
        seeds[f"{side}_putamen"] = putamen
    scale = float(data.max() - data.min())
    grown = growcut(data, seeds, scale)
    mask_path = output / "masks" / f"{uid}_standard_growcut.nii.gz"
    nib.save(nib.Nifti1Image(grown, image.affine, image.header), str(mask_path))
    return {
        "uid": uid,
        "is_pathologic": label,
        "left_caudate_voxels": int((grown == pipeline.ROI_LABELS["left_caudate"]).sum()),
        "left_putamen_voxels": int((grown == pipeline.ROI_LABELS["left_putamen"]).sum()),
        "right_caudate_voxels": int((grown == pipeline.ROI_LABELS["right_caudate"]).sum()),
        "right_putamen_voxels": int((grown == pipeline.ROI_LABELS["right_putamen"]).sum()),
        "mask_file": str(mask_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    labels = pipeline.load_labels(args.labels)
    paths = sorted(path for path in args.registered_dir.glob("*.nii.gz") if path.stem.removesuffix(".nii") in labels)
    paths = paths[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "masks").mkdir(exist_ok=True)
    payloads = [(str(path), labels[path.stem.removesuffix(".nii")], str(args.output_dir)) for path in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(process_one, payload) for payload in payloads]
        for number, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if number % 25 == 0 or number == len(futures):
                print(f"completed={number}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: str(row["uid"]))
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "README.md").write_text(
        f"# Standard Grow from Seeds baseline\n\nProcessed {len(rows)} ANTs-registered scans. Four automatically placed seed labels competed over the full image volume using intensity similarity. No search mask, background seeds, custom threshold, radius limit, or post-processing was applied.\n",
        encoding="utf-8",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
