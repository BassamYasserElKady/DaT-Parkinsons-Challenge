#!/usr/bin/env python3
"""Parallel full-cohort GrowCut-style segmentation on ANTs-registered scans."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

import growcut_registered_striatal_validation as gc
import validate_atlas_free_striatal_segmentation as pipeline


def process_one(payload: tuple[str, int, str, bool]) -> dict[str, object]:
    path_string, label, output_string, write_figure = payload
    path, output = Path(path_string), Path(output_string)
    uid = path.stem.removesuffix(".nii")
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0)
    brain = pipeline.brain_mask(data)
    bounds = pipeline.bounding_box(brain)
    midline = pipeline.find_midline(data, brain, bounds)
    allowed = pipeline.central_seed_zone(brain, bounds, midline, "left") | pipeline.central_seed_zone(brain, bounds, midline, "right")
    seeds = {}
    spacing = np.asarray(image.header.get_zooms()[:3])
    for side in ("left", "right"):
        zone = pipeline.central_seed_zone(brain, bounds, midline, side)
        caudate, putamen = pipeline.select_seed_pair(data, zone, side, spacing, 12.0, (2,))
        seeds[f"{side}_caudate"] = caudate
        seeds[f"{side}_putamen"] = putamen
    values = data[allowed]
    scale = max(float(np.percentile(values, 99.5) - np.percentile(values, 1.0)), 1e-6)
    grown = gc.growcut(data, allowed, seeds, scale)
    masks = {name: grown == label_id for name, label_id in pipeline.ROI_LABELS.items()}
    encoded = grown.copy()
    encoded[~allowed] = 0
    mask_path = output / "masks" / f"{uid}_growcut_rois.nii.gz"
    nib.save(nib.Nifti1Image(encoded, image.affine, image.header), str(mask_path))
    figure_path = output / "figures" / f"{uid}_growcut_rois.png"
    if write_figure:
        pipeline.save_figure(uid, data, masks, seeds, figure_path)
    return {
        "uid": uid, "is_pathologic": label, "midline_x": midline,
        "left_caudate_voxels": int(masks["left_caudate"].sum()),
        "left_putamen_voxels": int(masks["left_putamen"].sum()),
        "right_caudate_voxels": int(masks["right_caudate"].sum()),
        "right_putamen_voxels": int(masks["right_putamen"].sum()),
        "background_voxels": int((grown == gc.BACKGROUND_LABEL).sum()),
        "mask_file": str(mask_path), "figure_file": str(figure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--write-figures", action="store_true")
    args = parser.parse_args()
    labels = pipeline.load_labels(args.labels)
    paths = sorted(path for path in args.registered_dir.glob("*.nii.gz") if path.stem.removesuffix(".nii") in labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "masks").mkdir(exist_ok=True)
    if args.write_figures:
        (args.output_dir / "figures").mkdir(exist_ok=True)
    payloads = [(str(path), labels[path.stem.removesuffix(".nii")], str(args.output_dir), args.write_figures) for path in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(process_one, payload) for payload in payloads]
        for number, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if number % 25 == 0 or number == len(futures):
                print(f"completed={number}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: str(row["uid"]))
    with (args.output_dir / "growcut_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "README.md").write_text(
        "# Full-cohort GrowCut-style registered striatal masks\n\n"
        f"Processed {len(rows)} ANTs-registered scans. Four striatal labels competed against a background label within the central brain search region. "
        "This is a local GrowCut-style implementation, not the unavailable Slicer binary.\n",
        encoding="utf-8",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
