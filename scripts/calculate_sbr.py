#!/usr/bin/env python3
"""Calculate DAT-style specific binding ratios from MNI-registered NIfTI scans.

The atlas inputs are Harvard-Oxford maximum-probability maps on the same 2 mm
MNI grid as the registered scans.  The subcortical map supplies bilateral
caudate and putamen target ROIs; the cortical map supplies an occipital
reference ROI because no cerebellar atlas is currently bundled here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


SUB_LABELS = {
    5: "left_caudate",
    6: "left_putamen",
    # These indices refer to the raw Harvard-Oxford max-probability map.
    # The Nilearn symmetric-split view inserts a second brain-stem label,
    # which shifts the right-side labels by one; the saved NIfTI does not.
    16: "right_caudate",
    17: "right_putamen",
}

# Harvard-Oxford cortical labels containing an occipital component.  The
# atlas map values include background=0, so these are label-list indices.
OCCIPITAL_LABELS = {
    22: "lateral_occipital_superior",
    23: "lateral_occipital_inferior",
    39: "temporal_occipital_fusiform",
    40: "occipital_fusiform",
    48: "occipital_pole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--subcortical-atlas", type=Path, required=True)
    parser.add_argument("--cortical-atlas", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    return parser.parse_args()


def assert_same_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine):
        raise ValueError(
            f"{name} is not on the template grid: shape={other.shape}, "
            f"template_shape={reference.shape}"
        )


def mean_in_mask(data: np.ndarray, mask: np.ndarray) -> float:
    values = data[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def safe_sbr(target: float, reference: float) -> float:
    if not np.isfinite(target) or not np.isfinite(reference) or reference <= 0:
        return float("nan")
    return float((target - reference) / reference)


def safe_asymmetry(left: float, right: float) -> float:
    denominator = left + right
    if not np.isfinite(left) or not np.isfinite(right) or denominator == 0:
        return float("nan")
    return float(2.0 * abs(left - right) / denominator)


def main() -> None:
    args = parse_args()
    template = nib.load(str(args.template))
    sub_img = nib.load(str(args.subcortical_atlas))
    cort_img = nib.load(str(args.cortical_atlas))
    assert_same_grid(template, sub_img, "subcortical atlas")
    assert_same_grid(template, cort_img, "cortical atlas")

    sub = np.asarray(sub_img.dataobj)
    cort = np.asarray(cort_img.dataobj)
    masks = {name: sub == label for label, name in SUB_LABELS.items()}
    occipital = np.isin(cort, list(OCCIPITAL_LABELS))
    masks["occipital_reference"] = occipital

    if any(not mask.any() for mask in masks.values()):
        empty = [name for name, mask in masks.items() if not mask.any()]
        raise ValueError(f"Empty ROI mask(s): {empty}")

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    roi_mask = np.zeros(template.shape, dtype=np.int16)
    for index, name in enumerate(
        ["left_caudate", "right_caudate", "left_putamen", "right_putamen", "occipital_reference"],
        start=1,
    ):
        roi_mask[masks[name]] = index
    mask_header = template.header.copy()
    mask_header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(roi_mask, template.affine, mask_header), str(args.output_mask))

    labels = {}
    with args.labels.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["uid"]] = row["is_pathologic"]

    rows: list[dict[str, object]] = []
    scan_paths = sorted(args.registered_dir.glob("*.nii.gz"))
    for scan_path in scan_paths:
        uid = scan_path.name.removesuffix(".nii.gz")
        scan = nib.load(str(scan_path))
        assert_same_grid(template, scan, f"scan {uid}")
        data = np.asarray(scan.dataobj, dtype=np.float32)
        reference_mean = mean_in_mask(data, masks["occipital_reference"])
        means = {
            f"{name}_mean": mean_in_mask(data, masks[name])
            for name in SUB_LABELS.values()
        }
        left_caudate = means["left_caudate_mean"]
        right_caudate = means["right_caudate_mean"]
        left_putamen = means["left_putamen_mean"]
        right_putamen = means["right_putamen_mean"]
        caudate_mean = float(np.nanmean([left_caudate, right_caudate]))
        putamen_mean = float(np.nanmean([left_putamen, right_putamen]))
        striatum_mean = float(np.nanmean([left_caudate, right_caudate, left_putamen, right_putamen]))
        row: dict[str, object] = {
            "uid": uid,
            "is_pathologic": labels.get(uid, ""),
            "reference_mean": reference_mean,
            **means,
            "left_caudate_sbr": safe_sbr(left_caudate, reference_mean),
            "right_caudate_sbr": safe_sbr(right_caudate, reference_mean),
            "left_putamen_sbr": safe_sbr(left_putamen, reference_mean),
            "right_putamen_sbr": safe_sbr(right_putamen, reference_mean),
            "mean_caudate_sbr": safe_sbr(caudate_mean, reference_mean),
            "mean_putamen_sbr": safe_sbr(putamen_mean, reference_mean),
            "mean_striatum_sbr": safe_sbr(striatum_mean, reference_mean),
            "putamen_asymmetry": safe_asymmetry(left_putamen, right_putamen),
            "caudate_asymmetry": safe_asymmetry(left_caudate, right_caudate),
            "registered_image": str(scan_path),
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"No registered scans found in {args.registered_dir}")

    fieldnames = list(rows[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"scans={len(rows)}")
    print(f"output_csv={args.output_csv}")
    print(f"output_mask={args.output_mask}")
    print(f"reference_roi_voxels={int(occipital.sum())}")
    for name, mask in masks.items():
        print(f"{name}_voxels={int(mask.sum())}")


if __name__ == "__main__":
    main()
