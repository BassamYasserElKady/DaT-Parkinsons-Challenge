#!/usr/bin/env python3
"""Rigidly register NIfTI images in an archive to a reference template.

The input archive contains files named ``<uid>.nii.gz`` and the labels table
contains the corresponding ``uid`` values.  Registration is performed with a
3-D Euler transform and Mattes mutual information.  Each image is resampled
onto the template grid; the transform and a CSV audit log are saved alongside
the outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
import zipfile
from pathlib import Path

import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset in the labels table.")
    parser.add_argument("--summary-name", default="registration_summary.csv")
    parser.add_argument("--limit", type=int, default=None, help="Register only the first N rows.")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "uid" not in rows[0] or "is_pathologic" not in rows[0]:
        raise ValueError(f"Expected uid,is_pathologic columns in {path}")
    return rows


def register_one(moving_path: Path, fixed: sitk.Image) -> tuple[sitk.Image, sitk.Transform, float, int]:
    moving_raw = sitk.ReadImage(str(moving_path))
    moving = sitk.Cast(moving_raw, sitk.sitkFloat32)

    fixed_float = sitk.Cast(fixed, sitk.sitkFloat32)
    fixed_reg = sitk.Normalize(fixed_float)
    moving_reg = sitk.Normalize(moving)

    initial = sitk.CenteredTransformInitializer(
        fixed_reg,
        moving_reg,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.15)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=150,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)

    final_transform = registration.Execute(fixed_reg, moving_reg)
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return registered, final_transform, float(registration.GetMetricValue()), int(registration.GetOptimizerIteration())


def main() -> None:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    if not args.labels.exists():
        raise FileNotFoundError(args.labels)
    if not args.template.exists():
        raise FileNotFoundError(args.template)
    rows = read_labels(args.labels)
    if args.start < 0 or args.start > len(rows):
        raise ValueError(f"--start must be between 0 and {len(rows)}")
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = args.out_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / args.summary_name
    fixed = sitk.ReadImage(str(args.template))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, args.threads))

    with zipfile.ZipFile(args.archive) as archive:
        members = {Path(info.filename).stem.removesuffix(".nii"): info.filename for info in archive.infolist()}
        fieldnames = ["uid", "is_pathologic", "status", "metric", "iterations", "elapsed_sec", "registered_image", "transform", "error"]
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                uid = row["uid"]
                output_path = args.out_dir / f"{uid}.nii.gz"
                transform_path = transforms_dir / f"{uid}.tfm"
                result = {
                    "uid": uid,
                    "is_pathologic": row["is_pathologic"],
                    "status": "failed",
                    "metric": "",
                    "iterations": "",
                    "elapsed_sec": "",
                    "registered_image": str(output_path),
                    "transform": str(transform_path),
                    "error": "",
                }
                started = time.monotonic()
                try:
                    member = members.get(uid)
                    if member is None:
                        raise FileNotFoundError(f"No archive member for uid {uid}")
                    with tempfile.TemporaryDirectory(prefix="rigid_reg_") as temp_dir:
                        moving_path = Path(temp_dir) / Path(member).name
                        with archive.open(member) as source, moving_path.open("wb") as destination:
                            destination.write(source.read())
                        registered, transform, metric, iterations = register_one(moving_path, fixed)
                    sitk.WriteImage(registered, str(output_path), useCompression=True)
                    sitk.WriteTransform(transform, str(transform_path))
                    result.update({
                        "status": "ok",
                        "metric": metric,
                        "iterations": iterations,
                    })
                except Exception as exc:  # keep the batch auditable if one image is malformed
                    result["error"] = f"{type(exc).__name__}: {exc}"
                result["elapsed_sec"] = round(time.monotonic() - started, 3)
                writer.writerow(result)
                handle.flush()
                print(f"[{index}/{len(rows)}] {uid}: {result['status']} ({result['elapsed_sec']} s)", flush=True)


if __name__ == "__main__":
    main()
