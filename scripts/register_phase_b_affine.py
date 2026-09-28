#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/niftis_original"
MASKS = ROOT / "results/ants_registration_percentile10_masked/cleaned_masks"
TEMPLATE = ROOT / "submission_src_ensemble/assets/template.nii"
FIXED_MASK = ROOT / "assets/atlases/fsl/masks/MNI152_T1_2mm_brain_mask.nii.gz"
INIT = ROOT / "results/phase_a_initialization.csv"
OUT = ROOT / "results/phase_b_affine_masked"

def one(payload):
    uid, tx, ty, tz = payload
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        fixed = sitk.Cast(sitk.ReadImage(str(TEMPLATE)), sitk.sitkFloat32)
        moving = sitk.Cast(sitk.ReadImage(str(RAW / f"{uid}.nii.gz")), sitk.sitkFloat32)
        fixed_mask = sitk.Cast(sitk.ReadImage(str(FIXED_MASK)), sitk.sitkUInt8)
        moving_mask = sitk.Cast(sitk.ReadImage(str(MASKS / f"{uid}.nii.gz")), sitk.sitkUInt8)
        fixed_n = sitk.Normalize(fixed); moving_n = sitk.Normalize(moving)
        # SimpleITK uses its own physical LPS convention internally. Let its
        # geometric initializer compute the physical translation from headers.
        initial = sitk.CenteredTransformInitializer(
            fixed_n, moving_n, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingStrategy(reg.NONE)  # dense sampling
        reg.SetMetricFixedMask(fixed_mask)
        reg.SetMetricMovingMask(moving_mask)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=250,
                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        reg.SetInitialTransform(initial, inPlace=False)
        transform = reg.Execute(fixed_n, moving_n)
        out_img = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        sitk.WriteImage(out_img, str(OUT / f"{uid}.nii.gz"), useCompression=True)
        sitk.WriteTransform(transform, str(OUT / "transforms" / f"{uid}.tfm"))
        return {"uid":uid,"status":"ok","metric":float(reg.GetMetricValue()),"iterations":int(reg.GetOptimizerIteration()),"error":""}
    except Exception as e:
        return {"uid":uid,"status":"failed","metric":"","iterations":"","error":f"{type(e).__name__}: {e}"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--jobs',type=int,default=max(1,min(16,os.cpu_count() or 1))); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'transforms').mkdir(exist_ok=True)
    init={r['uid']:r for r in csv.DictReader(INIT.open(newline=''))}
    payload=[]
    for uid in sorted(init):
        if not (OUT/f'{uid}.nii.gz').exists() or not (OUT/'transforms'/f'{uid}.tfm').exists():
            r=init[uid]; payload.append((uid,r['translation_x_mm'],r['translation_y_mm'],r['translation_z_mm']))
    print(f'pending={len(payload)} total={len(init)} jobs={a.jobs}',flush=True)
    summary=OUT/'registration_summary.csv'; existing={}
    if summary.exists():
        for r in csv.DictReader(summary.open(newline='')): existing[r['uid']]=r
    fields=['uid','status','metric','iterations','elapsed_sec','error']
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        fs={pool.submit(one,p):p for p in payload}
        for i,f in enumerate(as_completed(fs),1):
            start=time.monotonic(); r=f.result(); r['elapsed_sec']=round(time.monotonic()-start,3); existing[r['uid']]=r
            if i%10==0 or i==len(payload): print(f"completed={i}/{len(payload)} uid={r['uid']} status={r['status']}",flush=True)
            with summary.open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(existing[k] for k in sorted(existing))
    print(f'output={OUT}',flush=True)

if __name__=='__main__':main()
