#!/usr/bin/env python3
"""Run the atlas-free seed-growth validator on ANTs-registered NIfTI files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

import validate_atlas_free_striatal_segmentation as pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--putamen-z-offsets", default="2")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--recent-ring-ratio", type=float, default=0.85)
    parser.add_argument("--min-seed-separation-mm", type=float, default=12.0)
    parser.add_argument("--max-physical-radius-mm", type=float, default=25.0)
    args = parser.parse_args()
    offsets = tuple(int(value.strip()) for value in args.putamen_z_offsets.split(",") if value.strip())
    labels = pipeline.load_labels(args.labels)
    paths = sorted(args.registered_dir.glob("*.nii.gz"))
    available = [path for path in paths if path.stem.removesuffix(".nii") in labels]
    if args.sample_size > len(available):
        raise ValueError(f"sample_size={args.sample_size} exceeds {len(available)} labeled scans")
    y = np.asarray([labels[path.stem.removesuffix(".nii")] for path in available])
    if args.sample_size == len(available):
        selected_paths = available
    else:
        _, selected = train_test_split(np.arange(len(available)), test_size=args.sample_size, stratify=y, random_state=args.seed)
        selected_paths = [available[index] for index in sorted(selected)]
    output = args.output_dir
    figure_dir, mask_dir = output / "figures", output / "masks"
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
        masks, seeds = {}, {}
        spacing = np.asarray(image.header.get_zooms()[:3])
        for side in ("left", "right"):
            zone = pipeline.central_seed_zone(brain, bounds, midline, side)
            caudate_seed, putamen_seed = pipeline.select_seed_pair(data, zone, side, spacing, args.min_seed_separation_mm, offsets)
            caudate, putamen = pipeline.grow_pair(
                data, brain, caudate_seed, putamen_seed, spacing, args.threshold,
                args.max_physical_radius_mm, args.recent_ring_ratio,
            )
            for structure, points, seed_point in (("caudate", caudate, caudate_seed), ("putamen", putamen, putamen_seed)):
                name = f"{side}_{structure}"
                masks[name] = pipeline.as_mask(data.shape, points)
                seeds[name] = seed_point
        encoded = np.zeros(data.shape, dtype=np.uint8)
        for name, label in pipeline.ROI_LABELS.items():
            encoded[masks[name]] = label
        mask_path = mask_dir / f"{uid}_atlas_free_rois.nii.gz"
        figure_path = figure_dir / f"{uid}_atlas_free_rois.png"
        nib.save(nib.Nifti1Image(encoded, image.affine, image.header), str(mask_path))
        pipeline.save_figure(uid, data, masks, seeds, figure_path)
        rows.append({
            "uid": uid, "is_pathologic": labels[uid], "midline_x": midline,
            "putamen_z_offsets": ",".join(map(str, offsets)),
            **{f"{name}_seed": ",".join(map(str, seeds[name])) for name in pipeline.ROI_LABELS},
            **{f"{name}_voxels": int(masks[name].sum()) for name in pipeline.ROI_LABELS},
            "mask_file": str(mask_path), "figure_file": str(figure_path),
        })
        print(f"[{number}/{len(selected_paths)}] {uid}", flush=True)
    with (output / "validation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.md").write_text(
        "# ANTs-registered atlas-free striatal validation\n\n"
        f"- Registered scans: {len(paths)}\n- Validation sample: {len(rows)}\n"
        f"- Putamen seed z offset: {','.join(map(str, offsets))}\n"
        "- Input: ANTs-registered volumes; no anatomical atlas labels were used.\n",
        encoding="utf-8",
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
