#!/usr/bin/env python3
"""Render selected slice windows and 0.15% hotspot overlays for download."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niftis-zip", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.selection_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(args.seed)
    selected = [rows[i] for i in sorted(rng.choice(len(rows), size=min(args.sample_size, len(rows)), replace=False))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_rows = []

    with zipfile.ZipFile(args.niftis_zip) as archive:
        for number, row in enumerate(selected, start=1):
            uid = row["uid"]
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted = Path(archive.extract(f"{uid}.nii.gz", temp_dir))
                data = np.asarray(nib.load(str(extracted)).dataobj, dtype=np.float32)
            data = np.where(np.isfinite(data), np.maximum(data, 0), 0)
            threshold = float(row["threshold_99_85"])
            start = int(row["start_zero_based"])
            end = int(row["end_zero_based"])
            z0, z1 = max(0, start - 2), min(data.shape[2] - 1, end + 2)
            slices = list(range(z0, z1 + 1))
            vmin, vmax = np.percentile(data, [1, 99.9])

            fig, axes = plt.subplots(1, len(slices), figsize=(2.1 * len(slices), 2.8), squeeze=False)
            for axis, z in zip(axes[0], slices):
                image = np.clip((data[:, :, z] - vmin) / max(vmax - vmin, 1e-6), 0, 1)
                axis.imshow(image.T, cmap="gray", origin="lower")
                hotspot = (data[:, :, z] > threshold).T
                overlay = np.zeros((*hotspot.shape, 4), dtype=float)
                overlay[hotspot] = (1.0, 0.0, 0.0, 0.85)
                axis.imshow(overlay, origin="lower")
                axis.set_title(f"z={z + 1}" + ("*" if start <= z <= end else ""), fontsize=8)
                axis.axis("off")
            fig.suptitle(f"{uid} | * = selected 17% window | red = top 0.15%", fontsize=10)
            fig.tight_layout()
            output = args.output_dir / f"{uid}_adaptive_window.png"
            fig.savefig(output, dpi=160, bbox_inches="tight")
            plt.close(fig)
            figure_rows.append({**row, "figure_file": str(output), "display_start_one_based": z0 + 1, "display_end_one_based": z1 + 1})
            print(f"completed={number}/{len(selected)} {uid}", flush=True)

    with (args.output_dir / "selected_figures_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_rows[0]))
        writer.writeheader()
        writer.writerows(figure_rows)
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
