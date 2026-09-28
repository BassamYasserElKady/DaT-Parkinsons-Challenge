#!/usr/bin/env python3
"""Register smoothed moving images while resampling the original intensities."""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def register_one(payload):
    uid, moving_path, template_path, output_path, transform_path, sigma = payload
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        moving_raw = sitk.ReadImage(moving_path)
        moving = sitk.Cast(moving_raw, sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(moving)
        smoothed_array = ndimage.gaussian_filter(array, sigma=float(sigma), mode="nearest")
        moving_smoothed = sitk.GetImageFromArray(smoothed_array.astype(np.float32))
        moving_smoothed.CopyInformation(moving)
        fixed = sitk.ReadImage(template_path)
        fixed_reg = sitk.Normalize(sitk.Cast(fixed, sitk.sitkFloat32))
        moving_reg = sitk.Normalize(moving_smoothed)
        initial = sitk.CenteredTransformInitializer(fixed_reg, moving_reg, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.15)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=150, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial, inPlace=False)
        transform = registration.Execute(fixed_reg, moving_reg)
        registered_original = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        sitk.WriteImage(registered_original, output_path, useCompression=True)
        sitk.WriteTransform(transform, transform_path)
        return {"uid": uid, "status": "ok", "metric": float(registration.GetMetricValue()), "iterations": int(registration.GetOptimizerIteration()), "error": ""}
    except Exception as exc:
        return {"uid": uid, "status": "failed", "metric": "", "iterations": "", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.75)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/external/niftis_original"
    template = root / "submission_src_ensemble/assets/template.nii"
    labels_path = root / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
    output_dir = root / "results/ants_registration_smoothed_optimizer_original_input"
    transform_dir = output_dir / "transforms"
    output_dir.mkdir(parents=True, exist_ok=True); transform_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(labels_path.open(newline="", encoding="utf-8")))
    payloads = []
    for row in rows:
        uid = row["uid"]
        output = output_dir / f"{uid}.nii.gz"
        transform = transform_dir / f"{uid}.tfm"
        if output.exists() and transform.exists():
            continue
        payloads.append((uid, str(raw / f"{uid}.nii.gz"), str(template), str(output), str(transform), args.sigma))
    print(f"pending={len(payloads)} total={len(rows)} jobs={args.jobs} sigma={args.sigma}", flush=True)
    summary = output_dir / "registration_summary.csv"
    existing = {}
    if summary.exists():
        for row in csv.DictReader(summary.open(newline="", encoding="utf-8")): existing[row["uid"]] = row
    fields = ["uid", "status", "metric", "iterations", "elapsed_sec", "registered_image", "transform", "error"]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(register_one, payload): payload for payload in payloads}
        for number, future in enumerate(as_completed(futures), 1):
            payload = futures[future]; started = time.monotonic(); result = future.result()
            result.update({"elapsed_sec": round(time.monotonic() - started, 3), "registered_image": payload[3], "transform": payload[4]})
            existing[result["uid"]] = result
            if number % 10 == 0 or number == len(payloads):
                print(f"completed={number}/{len(payloads)} uid={result['uid']} status={result['status']}", flush=True)
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerows(existing[uid] for uid in sorted(existing))
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__": main()
