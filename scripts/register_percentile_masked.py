#!/usr/bin/env python3
"""Register scans with per-scan nonzero 10th-percentile moving masks."""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def one(payload):
    uid, moving_path, template_path, fixed_mask_path, output_path, transform_path, prior_path = payload
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        moving_raw = sitk.ReadImage(moving_path)
        moving = sitk.Cast(moving_raw, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(moving)
        valid = np.isfinite(arr) & (arr > 0)
        values = arr[valid]
        if values.size == 0:
            raise ValueError("no positive voxels")
        cutoff = float(np.percentile(values, 10.0))
        mask_arr = (valid & (arr >= cutoff)).astype(np.uint8)
        moving_mask = sitk.GetImageFromArray(mask_arr)
        moving_mask.CopyInformation(moving)
        fixed = sitk.Cast(sitk.ReadImage(template_path), sitk.sitkFloat32)
        fixed_mask = sitk.Cast(sitk.ReadImage(fixed_mask_path), sitk.sitkUInt8)
        fixed_reg = sitk.Normalize(fixed)
        moving_reg = sitk.Normalize(moving)
        if prior_path and Path(prior_path).exists():
            initial = sitk.ReadTransform(prior_path)
        else:
            initial = sitk.CenteredTransformInitializer(
                fixed_reg, moving_reg, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.15)
        reg.SetMetricFixedMask(fixed_mask)
        reg.SetMetricMovingMask(moving_mask)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=150,
                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(initial, inPlace=False)
        transform = reg.Execute(fixed_reg, moving_reg)
        registered = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        sitk.WriteImage(registered, output_path, useCompression=True)
        sitk.WriteTransform(transform, transform_path)
        return {"uid": uid, "status": "ok", "cutoff": cutoff,
                "positive_voxels": int(values.size), "mask_voxels": int(mask_arr.sum()),
                "metric": float(reg.GetMetricValue()), "iterations": int(reg.GetOptimizerIteration()), "error": ""}
    except Exception as exc:
        return {"uid": uid, "status": "failed", "cutoff": "", "positive_voxels": "",
                "mask_voxels": "", "metric": "", "iterations": "",
                "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/external/niftis_original"
    template = root / "submission_src_ensemble/assets/template.nii"
    fixed_mask = root / "assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz"
    out = root / "results/ants_registration_percentile10_masked"
    out.mkdir(parents=True, exist_ok=True)
    (out / "transforms").mkdir(exist_ok=True)
    rows = list(csv.DictReader((root / "results/ants_registered_consecutive_resnet18_cbam/split.csv").open(newline="")))
    payloads = []
    for row in rows:
        uid = row["uid"]
        image = out / f"{uid}.nii.gz"
        transform = out / "transforms" / f"{uid}.tfm"
        if not image.exists() or not transform.exists():
            payloads.append((uid, str(raw / f"{uid}.nii.gz"), str(template), str(fixed_mask), str(image), str(transform), str(root / "results/ants_registration_mni/transforms" / f"{uid}.mat")))
    print(f"pending={len(payloads)} total={len(rows)} jobs={args.jobs}", flush=True)
    summary_path = out / "registration_summary.csv"
    existing = {}
    if summary_path.exists():
        for r in csv.DictReader(summary_path.open(newline="")): existing[r["uid"]] = r
    fields = ["uid", "status", "cutoff", "positive_voxels", "mask_voxels", "metric", "iterations", "elapsed_sec", "error"]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(one, p): p for p in payloads}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]; start = time.monotonic(); result = fut.result()
            result["elapsed_sec"] = round(time.monotonic() - start, 3)
            existing[result["uid"]] = result
            if i % 10 == 0 or i == len(payloads):
                print(f"completed={i}/{len(payloads)} uid={result['uid']} status={result['status']}", flush=True)
            with summary_path.open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(existing[k] for k in sorted(existing))
    print(f"output={out}", flush=True)


if __name__ == "__main__":
    main()
