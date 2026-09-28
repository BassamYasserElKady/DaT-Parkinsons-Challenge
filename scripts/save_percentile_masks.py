#!/usr/bin/env python3
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/niftis_original"
OUT = ROOT / "results/ants_registration_percentile10_masked/masks"
ROWS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"

def one(uid):
    src = RAW / f"{uid}.nii.gz"; dst = OUT / f"{uid}.nii.gz"
    if dst.exists(): return uid
    img = sitk.ReadImage(str(src)); a = sitk.GetArrayFromImage(img)
    v = np.isfinite(a) & (a > 0); cutoff = np.percentile(a[v], 10.0)
    m = sitk.GetImageFromArray((v & (a >= cutoff)).astype(np.uint8)); m.CopyInformation(img)
    sitk.WriteImage(m, str(dst), useCompression=True)
    return uid

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    uids = [r["uid"] for r in csv.DictReader(ROWS.open(newline=""))]
    with ProcessPoolExecutor(max_workers=16) as pool:
        fs = [pool.submit(one, u) for u in uids]
        for i, f in enumerate(as_completed(fs), 1):
            f.result()
            if i % 100 == 0 or i == len(fs): print(f"saved={i}/{len(fs)}", flush=True)
    print(OUT)

if __name__ == "__main__": main()
