#!/usr/bin/env python3
"""QC screen for ring-like radial patterns and clean percentile masks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "results/ants_registration_mni"
MASKS = ROOT / "results/ants_registration_percentile10_masked/masks"
OUT = ROOT / "results/ants_registration_percentile10_masked/cleaned_masks"

STRUCT = np.ones((3, 3, 3), dtype=np.uint8)

def ring_score(vol):
    # Heuristic candidate screen: persistent sharp radial bands in registered
    # axial slices. This is a QC score, not a diagnostic label.
    scores = []
    nx, ny, nz = vol.shape
    xx, yy = np.mgrid[:nx, :ny]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    bins = np.arange(4, min(nx, ny) / 2.0, 1.0)
    for z in range(nz):
        sl = np.asarray(vol[:, :, z], dtype=np.float32)
        valid = np.isfinite(sl) & (sl > 0)
        if valid.sum() < 100:
            continue
        prof = []
        for lo in bins:
            sel = valid & (rr >= lo) & (rr < lo + 1)
            prof.append(float(np.mean(sl[sel])) if sel.sum() > 10 else np.nan)
        prof = np.asarray(prof)
        good = np.isfinite(prof)
        if good.sum() < 12:
            continue
        prof = np.interp(np.arange(len(prof)), np.flatnonzero(good), prof[good])
        smooth = ndimage.gaussian_filter1d(prof, 2.0)
        residual = prof - smooth
        scale = np.median(np.abs(residual - np.median(residual))) + 1e-6
        scores.append(float(np.percentile(np.abs(residual), 95) / scale))
    return float(np.percentile(scores, 90)) if scores else 0.0

def one(uid):
    image_path = IMAGES / f"{uid}.nii.gz"
    mask_path = MASKS / f"{uid}.nii.gz"
    out_path = OUT / f"{uid}.nii.gz"
    img = nib.load(str(image_path)); vol = np.asanyarray(img.dataobj)
    raw_img = nib.load(str(ROOT / "data/external/niftis_original" / f"{uid}.nii.gz"))
    m = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
    labels, count = ndimage.label(m, structure=STRUCT)
    sizes = np.bincount(labels.ravel())
    if len(sizes) > 1:
        largest = 1 + int(np.argmax(sizes[1:]))
        lcc = labels == largest
    else:
        lcc = m
    filled = ndimage.binary_fill_holes(lcc)
    nib.save(nib.Nifti1Image(filled.astype(np.uint8), raw_img.affine, raw_img.header), str(out_path))
    score = ring_score(vol)
    return {"uid": uid, "components_before": int(count),
            "mask_voxels_before": int(m.sum()), "lcc_voxels": int(lcc.sum()),
            "cleaned_voxels": int(filled.sum()),
            "lcc_fraction": float(lcc.sum() / max(m.sum(), 1)),
            "hole_fill_added": int(filled.sum() - lcc.sum()),
            "ring_score": score}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    uids = sorted(p.name[:-7] for p in MASKS.glob("*.nii.gz"))
    rows = []
    with ProcessPoolExecutor(max_workers=16) as pool:
        fs = [pool.submit(one, uid) for uid in uids]
        for i, f in enumerate(as_completed(fs), 1):
            rows.append(f.result())
            if i % 100 == 0 or i == len(fs): print(f"processed={i}/{len(fs)}", flush=True)
    rows.sort(key=lambda x: x["uid"])
    report = ROOT / "results/ants_registration_percentile10_masked/mask_cleanup_ring_qc.csv"
    with report.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    scores = np.asarray([r["ring_score"] for r in rows])
    threshold = float(np.percentile(scores, 99))
    candidates = [r for r in rows if r["ring_score"] >= threshold]
    with (ROOT / "results/ants_registration_percentile10_masked/ring_artifact_candidates.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["uid", "ring_score", "threshold", "candidate_rule"]); w.writeheader()
        for r in sorted(candidates, key=lambda x: -x["ring_score"]):
            w.writerow({"uid": r["uid"], "ring_score": r["ring_score"], "threshold": threshold, "candidate_rule": "top 1% heuristic radial-band score"})
    print(f"scans={len(rows)}")
    print(f"mean_components_before={np.mean([r['components_before'] for r in rows]):.3f}")
    print(f"mean_lcc_fraction={np.mean([r['lcc_fraction'] for r in rows]):.4f}")
    print(f"scans_hole_fill_added={sum(r['hole_fill_added'] > 0 for r in rows)}")
    print(f"ring_score_threshold_top1pct={threshold:.4f}")
    print(f"ring_artifact_candidates={len(candidates)}")
    print(f"report={report}")

if __name__ == "__main__": main()
