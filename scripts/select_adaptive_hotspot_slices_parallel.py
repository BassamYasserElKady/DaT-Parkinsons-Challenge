#!/usr/bin/env python3
"""Parallel runner for adaptive high-intensity slice selection."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from select_adaptive_hotspot_slices import choose_window, load_labels


def process_one(payload: tuple[str, int, str, float, str]) -> dict[str, object]:
    uid, label, archive_path, fraction, output_dir = payload
    output = Path(output_dir)
    with zipfile.ZipFile(archive_path) as archive:
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted = Path(archive.extract(f"{uid}.nii.gz", temp_dir))
            image = nib.load(str(extracted))
            data = np.asarray(image.dataobj, dtype=np.float32)
            selection = choose_window(data, fraction, leading_discard_fraction=0.04)
            start = int(selection["start_zero_based"])
            end = int(selection["end_zero_based"]) + 1
            selected = np.where(np.isfinite(data[:, :, start:end]), np.maximum(data[:, :, start:end], 0), 0).astype(np.float32)
            affine = image.affine.copy()
            affine[:3, 3] += affine[:3, 2] * start
            header = image.header.copy()
            header.set_data_shape(selected.shape)
            output_path = output / "selected_slices" / f"{uid}_selected_{selection['selected_slices']}of{selection['eligible_nonempty_slices']}.nii.gz"
            nib.save(nib.Nifti1Image(selected, affine, header), str(output_path))
    return {"uid": uid, "is_pathologic": label, **selection, "selected_file": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niftis-zip", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=7 / 41)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    labels = load_labels(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_slices").mkdir(exist_ok=True)
    with zipfile.ZipFile(args.niftis_zip) as archive:
        available = {Path(name).stem.removesuffix(".nii") for name in archive.namelist() if name.endswith(".nii.gz")}
    uids = sorted(set(labels) & available)
    payloads = [(uid, labels[uid], str(args.niftis_zip), args.fraction, str(args.output_dir)) for uid in uids]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(process_one, payload) for payload in payloads]
        for number, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if number % 50 == 0 or number == len(futures):
                print(f"completed={number}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: str(row["uid"]))
    with (args.output_dir / "adaptive_slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "README.md").write_text(
        "# Adaptive 0.15% hotspot slice selection with leading discard\n\n"
        f"Processed {len(rows)} raw, unregistered scans. The leading 4% of nonempty slices was discarded, then a contiguous block of approximately 7/41 of the remaining nonempty slices was selected by largest 26-connected hotspot component.\n",
        encoding="utf-8",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
