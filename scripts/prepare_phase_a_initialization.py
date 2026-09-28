#!/usr/bin/env python3
from pathlib import Path
import csv
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/niftis_original"
TEMPLATE = ROOT / "submission_src_ensemble/assets/template.nii"
OUT = ROOT / "results/phase_a_initialization.csv"

def world_center(img):
    idx = np.array([(n - 1) / 2.0 for n in img.shape[:3]] + [1.0])
    return img.affine @ idx

def main():
    fixed = nib.load(str(TEMPLATE))
    fc = world_center(fixed)[:3]
    rows = []
    for p in sorted(RAW.glob("*.nii.gz")):
        img = nib.load(str(p)); mc = world_center(img)[:3]
        t = fc - mc
        rows.append({"uid": p.name[:-7], "orientation": "".join(nib.aff2axcodes(img.affine)),
                     "fixed_cx_mm": fc[0], "fixed_cy_mm": fc[1], "fixed_cz_mm": fc[2],
                     "moving_cx_mm": mc[0], "moving_cy_mm": mc[1], "moving_cz_mm": mc[2],
                     "translation_x_mm": t[0], "translation_y_mm": t[1], "translation_z_mm": t[2],
                     "translation_norm_mm": float(np.linalg.norm(t))})
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    norms = np.array([r["translation_norm_mm"] for r in rows])
    print(f"scans={len(rows)}")
    print(f"fixed_center_mm={fc.tolist()}")
    print(f"translation_norm_median_mm={np.median(norms):.3f}")
    print(f"translation_norm_min_max_mm={norms.min():.3f},{norms.max():.3f}")
    print(f"orientations={sorted(set(r['orientation'] for r in rows))}")
    print(f"output={OUT}")

if __name__ == "__main__": main()
