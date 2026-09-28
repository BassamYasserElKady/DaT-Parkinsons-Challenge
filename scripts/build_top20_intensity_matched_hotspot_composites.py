#!/usr/bin/env python3
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results/ants_registration_mni"
PRED = ROOT / "results/cv5_real8_post_registration_gaussian/oof_predictions.csv"
TEMPLATE_PATH = ROOT / "submission_src_ensemble/assets/template.nii"
OUT = ROOT / "results/top20_intensity_matched_hotspot_composites"
UIDS = ["bmae38sr","7875aud9","xid4n1u3","y08p0fod","uh17j8qo","bglmy62x","c1m3o55i","rqq5qob8","07gbasf5","vfua9bq1","0imjyjcm","344mw04i","0uc87nxp","l5u7nted","nal3gywi","nkb4ehou","6tx0tbkg","ccprhvlz","f5gzpl1q","dhvs41lq"]

def top_mask(a, percent):
    positive = a[np.isfinite(a) & (a > 0)]
    cutoff = float(np.percentile(positive, 100.0 - percent))
    return np.isfinite(a) & (a >= cutoff), cutoff

def normalized(points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (points - lo) / np.maximum(hi - lo, 1)

def rank_match(values, reference):
    order = np.argsort(values, kind="mergesort")
    target = np.quantile(reference, np.linspace(0, 1, len(values)))
    matched = np.empty(len(values), dtype=np.float32)
    matched[order] = target.astype(np.float32)
    return matched

def inject(template, scan, destination, source):
    output = template.copy()
    output[destination] = 0
    xgrid = np.indices(template.shape)[0]
    midpoint = template.shape[0] / 2.0
    for side in (xgrid < midpoint, xgrid >= midpoint):
        dpts = np.argwhere(destination & side)
        spts = np.argwhere(source & side)
        if not len(dpts) or not len(spts):
            continue
        nearest = cKDTree(normalized(spts)).query(normalized(dpts), k=1)[1]
        sampled_source = scan[tuple(spts[nearest].T)]
        template_reference = template[tuple(dpts.T)]
        output[tuple(dpts.T)] = rank_match(sampled_source, template_reference)
    return output

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timg = nib.load(str(TEMPLATE_PATH)); template = timg.get_fdata().astype(np.float32)
    destination, template_cutoff = top_mask(template, 1.0)
    removed = template.copy(); removed[destination] = 0
    predictions = pd.read_csv(PRED).set_index("uid")
    z_peak = int(np.argmax(destination.sum(axis=(0, 1))))
    slices = list(range(max(0, z_peak - 3), min(template.shape[2], z_peak + 5)))
    while len(slices) < 8: slices.append(slices[-1])
    vmax = float(np.percentile(template[template > 0], 99.5))
    for uid in UIDS:
        scan = nib.load(str(REG / f"{uid}.nii.gz")).get_fdata().astype(np.float32)
        source, scan_cutoff = top_mask(scan, 0.25)
        composite = inject(template, scan, destination, source)
        header = timg.header.copy(); header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(composite, timg.affine, header), str(OUT / f"{uid}_matched_composite.nii.gz"))
        fig, axes = plt.subplots(3, 8, figsize=(20, 9), constrained_layout=True)
        for col, z in enumerate(slices):
            for row, (vol, title) in enumerate(((template, "Original template"), (removed, "Top 1% removed"), (composite, "Matched hotspot inserted"))):
                axes[row, col].imshow(np.rot90(vol[:, :, z].T), cmap="gray", vmin=0, vmax=vmax)
                axes[row, col].set_title(f"{title}\nz={z}", fontsize=8)
                axes[row, col].axis("off")
        y = int(predictions.loc[uid, "actual"]); p = float(predictions.loc[uid, "equal_ensemble_clipped"])
        loss = -(y*np.log(np.clip(p,1e-15,1-1e-15)) + (1-y)*np.log(np.clip(1-p,1e-15,1-1e-15)))
        fig.suptitle(f"{uid}: top-0.25% geometry inserted after exact rank matching to template top-1% intensities | loss={loss:.3f} | source cutoff={scan_cutoff:.5g}", fontsize=12)
        fig.savefig(OUT / f"{uid}_matched_composite.png", dpi=150)
        plt.close(fig)
    zip_path = ROOT / "results/top20_intensity_matched_hotspot_composites.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()): archive.write(path, path.name)
    print(f"template_top1_cutoff={template_cutoff}")
    print(f"destination_voxels={int(destination.sum())}")
    print(f"composites={len(list(OUT.glob('*_matched_composite.nii.gz')))}")
    print(zip_path)

if __name__ == "__main__":
    main()
