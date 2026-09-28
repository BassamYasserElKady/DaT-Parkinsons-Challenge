#!/usr/bin/env python3

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "ants_registration_mni"
MASK_PATH = ROOT / "assets" / "atlases" / "fsl" / "masks" / "MNI152_T1_2mm_brain_mask.nii.gz"
OUTPUT_DIR = ROOT / "results" / "ants_rigid_brainmask_mode_normalized"
VOLUME_DIR = OUTPUT_DIR / "volumes"
SUMMARY_PATH = OUTPUT_DIR / "mode_normalization_qc.csv"
HISTOGRAM_BINS = 4096
WORKERS = min(16, os.cpu_count() or 1)


def process(scan_path_str):
    scan_path = Path(scan_path_str)
    uid = scan_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    output_path = VOLUME_DIR / f"{uid}.nii.gz"

    image = nib.load(str(scan_path))
    image_data = np.asarray(image.dataobj, dtype=np.float32)
    mask_image = nib.load(str(MASK_PATH))
    mask = np.asarray(mask_image.dataobj) > 0

    if image_data.shape != mask.shape or not np.allclose(image.affine, mask_image.affine, atol=1e-5):
        raise ValueError(f"{uid}: scan and brain-mask geometry differ")

    valid = mask & np.isfinite(image_data) & (image_data > 0)
    values = image_data[valid]
    if values.size == 0:
        raise ValueError(f"{uid}: no finite positive voxels inside brain mask")

    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS)
    mode_bin = int(np.argmax(counts))
    mode = float((edges[mode_bin] + edges[mode_bin + 1]) / 2.0)
    if not np.isfinite(mode) or mode <= 0:
        raise ValueError(f"{uid}: invalid mode {mode}")

    normalized = np.zeros(image_data.shape, dtype=np.float32)
    inside = mask & np.isfinite(image_data)
    normalized[inside] = image_data[inside] / mode

    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(normalized, image.affine, header), str(output_path))

    return {
        "uid": uid,
        "status": "completed",
        "brain_mask_voxels": int(mask.sum()),
        "positive_in_mask_voxels": int(values.size),
        "histogram_bins": HISTOGRAM_BINS,
        "mode_bin": mode_bin,
        "mode_intensity": mode,
        "mode_bin_count": int(counts[mode_bin]),
        "mode_after_scaling": 1.0,
        "input_min_positive": float(values.min()),
        "input_max": float(values.max()),
        "output_path": str(output_path),
        "error": "",
    }


def write_summary(rows):
    fields = [
        "uid", "status", "brain_mask_voxels", "positive_in_mask_voxels",
        "histogram_bins", "mode_bin", "mode_intensity", "mode_bin_count",
        "mode_after_scaling", "input_min_positive", "input_max", "output_path", "error",
    ]
    temporary = SUMMARY_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["uid"]))
    temporary.replace(SUMMARY_PATH)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    scans = sorted(INPUT_DIR.glob("*.nii.gz")) + sorted(INPUT_DIR.glob("*.nii"))
    print(
        f"scans={len(scans)} workers={WORKERS} bins={HISTOGRAM_BINS} "
        f"mask={MASK_PATH}",
        flush=True,
    )

    rows = []
    failures = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process, str(path)): path for path in scans}
        for index, future in enumerate(as_completed(futures), start=1):
            scan_path = futures[future]
            uid = scan_path.name.removesuffix(".nii.gz").removesuffix(".nii")
            try:
                row = future.result()
            except Exception as error:
                failures += 1
                row = {
                    "uid": uid, "status": "failed", "brain_mask_voxels": "",
                    "positive_in_mask_voxels": "", "histogram_bins": HISTOGRAM_BINS,
                    "mode_bin": "", "mode_intensity": "", "mode_bin_count": "",
                    "mode_after_scaling": "", "input_min_positive": "", "input_max": "",
                    "output_path": "", "error": str(error),
                }
            rows.append(row)
            if index % 25 == 0 or row["status"] == "failed" or index == len(scans):
                write_summary(rows)
                print(f"[{index}/{len(scans)}] failures={failures} last={uid} {row['status']}", flush=True)

    print(f"finished completed={len(rows) - failures} failed={failures}", flush=True)


if __name__ == "__main__":
    main()
