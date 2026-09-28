#!/usr/bin/env python3
"""Validate top-90%-of-positive-voxels mask construction."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def validate_one(image_path: Path, mask_path: Path | None = None):
    image_data = np.asanyarray(nib.load(str(image_path)).dataobj)
    pos = image_data[image_data > 0]
    if len(pos) == 0:
        raise ValueError(f"no positive voxels: {image_path}")
    cutoff = float(np.percentile(pos, 10))
    below = int(np.sum(pos < cutoff))
    tied = int(np.sum(pos == cutoff))
    kept = int(np.sum(pos >= cutoff))
    mask = (image_data >= cutoff).astype(np.uint8)
    retained_positive = int(np.sum((image_data >= cutoff) & (image_data > 0)))
    zeros = int(np.sum(image_data == 0))
    total = int(image_data.size)
    saved_match = "not_checked"
    saved_ones = ""
    if mask_path is not None and mask_path.exists():
        saved = np.asanyarray(nib.load(str(mask_path)).dataobj)
        saved_match = bool(np.array_equal(mask, saved))
        saved_ones = int(np.sum(saved != 0))
    return {
        "uid": image_path.name.split(".")[0], "shape": "x".join(map(str, image_data.shape)),
        "total": total, "zero": zeros, "positive": len(pos), "cutoff": cutoff,
        "below": below, "tied": tied, "kept": kept,
        "retention": retained_positive / len(pos),
        "zero_mask_ones": int(np.sum((image_data == 0) & (mask == 1))),
        "saved_match": saved_match, "saved_ones": saved_ones,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/external/niftis_original"
    masks = root / "results/ants_registration_percentile10_masked/masks"
    if args.all or args.path is None:
        paths = sorted(raw.glob("*.nii.gz"))
    else:
        paths = [args.path]
    rows = []
    for i, path in enumerate(paths, 1):
        row = validate_one(path, masks / path.name)
        rows.append(row)
        if i % 200 == 0 or i == len(paths):
            print(f"validated={i}/{len(paths)}", flush=True)
    out = root / "results/ants_registration_percentile10_masked/mask_validation.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    r = rows[0]
    print("\nREPRESENTATIVE SCAN")
    print("field                         value")
    print(f"uid                           {r['uid']}")
    print(f"matrix shape                 {r['shape']}")
    print(f"total matrix voxels          {r['total']}")
    print(f"zero voxels                  {r['zero']} ({100*r['zero']/r['total']:.3f}%)")
    print(f"positive voxels              {r['positive']}")
    print(f"10th percentile cutoff       {r['cutoff']!r}")
    print(f"positive voxels < cutoff     {r['below']}")
    print(f"tied voxels == cutoff       {r['tied']}")
    print(f"positive voxels >= cutoff   {r['kept']}")
    print("nominal target retention    90.000%")
    print(f"empirical retention         {100*r['retention']:.3f}%")
    print(f"zero voxels labeled 1       {r['zero_mask_ones']}")
    print(f"saved mask exact match      {r['saved_match']}")
    print("\nALL-SCAN SUMMARY")
    print(f"scans validated             {len(rows)}")
    print(f"saved masks exact matches   {sum(x['saved_match'] is True for x in rows)}/{len(rows)}")
    print(f"mean empirical retention    {100*np.mean([x['retention'] for x in rows]):.3f}%")
    print(f"median empirical retention  {100*np.median([x['retention'] for x in rows]):.3f}%")
    print(f"min/max retention           {100*min(x['retention'] for x in rows):.3f}% / {100*max(x['retention'] for x in rows):.3f}%")
    print(f"mean tied-boundary voxels   {np.mean([x['tied'] for x in rows]):.1f}")
    print(f"zero-label violations       {sum(x['zero_mask_ones'] for x in rows)}")
    verdict = all(x['zero_mask_ones'] == 0 and x['saved_match'] is True for x in rows)
    print("\nVERDICT")
    print("PASS: saved masks conform to mask = (image_data >= np.percentile(image_data[image_data > 0], 10))" if verdict else "FAIL: at least one saved mask differs from the specified rule")
    print(f"report={out}")


if __name__ == "__main__":
    main()
