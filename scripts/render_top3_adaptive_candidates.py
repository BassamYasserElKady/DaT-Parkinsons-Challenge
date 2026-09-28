#!/usr/bin/env python3
"""Render three distinct adaptive hotspot-window candidates for random scans."""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


def candidates(data: np.ndarray, fraction: float = 7 / 41, discard_fraction: float = 0.04, count: int = 3):
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0).astype(np.float32, copy=False)
    threshold = float(np.percentile(data, 99.85))
    hot = data > threshold
    nonempty = np.flatnonzero(np.max(data, axis=(0, 1)) > 0)
    discarded = min(int(math.ceil(discard_fraction * nonempty.size)), nonempty.size - 1)
    eligible = nonempty[discarded:]
    window = max(1, min(eligible.size, int(round(fraction * eligible.size))))
    labels, component_count = ndimage.label(hot, structure=np.ones((3, 3, 3), dtype=np.uint8))
    per_slice = np.zeros((component_count, data.shape[2]), dtype=np.int32)
    for z in range(data.shape[2]):
        per_slice[:, z] = np.bincount(labels[:, :, z].ravel(), minlength=component_count + 1)[1:]
    scored = []
    for start in range(int(eligible[0]), int(eligible[-1]) - window + 2):
        cc = per_slice[:, start:start + window].sum(axis=1)
        scored.append({
            "start": start,
            "end": start + window - 1,
            "largest": int(cc.max()) if component_count else 0,
            "total": int(hot[:, :, start:start + window].sum()),
            "components": int(np.count_nonzero(cc)),
        })
    scored.sort(key=lambda r: (-r["largest"], -r["total"], r["start"]))
    chosen = []
    minimum_separation = max(1, window // 2)
    for row in scored:
        if all(abs(row["start"] - other["start"]) >= minimum_separation for other in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return data, hot, threshold, window, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niftis-zip", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.labels.open(newline="", encoding="utf-8") as handle:
        label_rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(args.seed)
    chosen_rows = [label_rows[i] for i in sorted(rng.choice(len(label_rows), size=min(args.sample_size, len(label_rows)), replace=False))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    with zipfile.ZipFile(args.niftis_zip) as archive:
        for number, label_row in enumerate(chosen_rows, start=1):
            uid = label_row["uid"]
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted = Path(archive.extract(f"{uid}.nii.gz", temp_dir))
                data = np.asarray(nib.load(str(extracted)).dataobj, dtype=np.float32)
            data, hot, threshold, window, rows = candidates(data)
            lo, hi = np.percentile(data, [1, 99.9])
            fig, axes = plt.subplots(len(rows), 1, figsize=(6, 3.7 * len(rows)), squeeze=False)
            for rank, (axis, row) in enumerate(zip(axes[:, 0], rows), start=1):
                start, end = row["start"], row["end"]
                projection = np.max(data[:, :, start:end + 1], axis=2).T
                hotspot = np.any(hot[:, :, start:end + 1], axis=2).T
                image = np.clip((projection - lo) / max(hi - lo, 1e-6), 0, 1)
                axis.imshow(image, cmap="gray", origin="lower")
                overlay = np.zeros((*hotspot.shape, 4), dtype=float)
                overlay[hotspot] = (1.0, 0.0, 0.0, 0.85)
                axis.imshow(overlay, origin="lower")
                axis.set_title(
                    f"Candidate {rank}: slices {start + 1}-{end + 1} | "
                    f"largest={row['largest']:,} | hotspots={row['total']:,}", fontsize=10
                )
                axis.axis("off")
                manifest.append({
                    "uid": uid, "is_pathologic": int(float(label_row["is_pathologic"])),
                    "rank": rank, "threshold_99_85": threshold,
                    "window_slices": window, "start_one_based": start + 1,
                    "end_one_based": end + 1, "largest_connected_hot_voxels": row["largest"],
                    "total_hot_voxels": row["total"], "components_in_window": row["components"],
                })
            fig.suptitle(f"{uid} | {'patient' if int(float(label_row['is_pathologic'])) else 'healthy'} | red = top 0.15%", fontsize=12)
            fig.tight_layout()
            fig.savefig(args.output_dir / f"{uid}_top3_candidates.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"completed={number}/{len(chosen_rows)} {uid}", flush=True)

    with (args.output_dir / "candidate_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
