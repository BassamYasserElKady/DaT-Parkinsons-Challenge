#!/usr/bin/env python3
"""Select one contiguous high-intensity slice block from every raw NIfTI scan."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path
from math import ceil

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["uid"]: int(float(row["is_pathologic"])) for row in csv.DictReader(handle)}


def choose_window(data: np.ndarray, fraction: float, reference_slices: int = 41, leading_discard_fraction: float = 0.04) -> dict[str, object]:
    if data.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {data.shape}")
    data = np.where(np.isfinite(data), np.maximum(data, 0), 0).astype(np.float32, copy=False)
    threshold = float(np.percentile(data, 99.85))
    hot = data > threshold
    n_slices = int(data.shape[2])
    nonempty_indices = np.flatnonzero(np.max(data, axis=(0, 1)) > 0)
    if nonempty_indices.size == 0:
        raise ValueError("scan has no nonempty slices")
    discarded = min(int(ceil(leading_discard_fraction * nonempty_indices.size)), nonempty_indices.size - 1)
    eligible_indices = nonempty_indices[discarded:]
    eligible_count = int(eligible_indices.size)
    window = max(1, min(eligible_count, int(round(fraction * eligible_count))))
    # 7/41 is the requested reference ratio; use the same ratio after discarding.
    if reference_slices == 41 and abs(fraction - 7 / 41) < 1e-12:
        window = max(1, min(eligible_count, int(round(7 * eligible_count / 41))))

    connectivity = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connected in 3D
    # Label once in 3D, then count each component inside each candidate window.
    # This preserves global 26-connectivity while avoiding a full relabel per window.
    component_labels, component_count = ndimage.label(hot, structure=connectivity)
    per_slice_counts = np.zeros((component_count, n_slices), dtype=np.int32)
    for z in range(n_slices):
        per_slice_counts[:, z] = np.bincount(
            component_labels[:, :, z].ravel(), minlength=component_count + 1
        )[1:]

    best: tuple[int, int, int, int] | None = None
    first_eligible = int(eligible_indices[0])
    last_eligible = int(eligible_indices[-1])
    for start in range(first_eligible, last_eligible - window + 2):
        component_counts = per_slice_counts[:, start : start + window].sum(axis=1)
        largest = int(component_counts.max()) if component_count else 0
        total_hot = int(hot[:, :, start : start + window].sum())
        components = int(np.count_nonzero(component_counts))
        # Largest connected region first, then total hotspots, then earliest window.
        candidate = (largest, total_hot, -start, components)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise ValueError("could not evaluate a slice window")
    largest, total_hot, negative_start, components = best
    start = -negative_start
    return {
        "threshold_99_85": threshold,
        "n_slices": n_slices,
        "nonempty_slices": int(nonempty_indices.size),
        "discarded_leading_nonempty_slices": discarded,
        "eligible_nonempty_slices": eligible_count,
        "selected_slices": window,
        "start_zero_based": start,
        "end_zero_based": start + window - 1,
        "start_one_based": start + 1,
        "end_one_based": start + window,
        "largest_connected_hot_voxels": largest,
        "total_hot_voxels": total_hot,
        "components_in_window": components,
        "hot_voxels_total": int(hot.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niftis-zip", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=7 / 41)
    parser.add_argument("--leading-discard-fraction", type=float, default=0.04)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = args.output_dir / "selected_slices"
    selected_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    with zipfile.ZipFile(args.niftis_zip) as archive:
        available = {Path(name).stem.removesuffix(".nii") for name in archive.namelist() if name.endswith(".nii.gz")}
        uids = sorted(set(labels) & available)
        for number, uid in enumerate(uids, start=1):
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted = Path(archive.extract(f"{uid}.nii.gz", temp_dir))
                image = nib.load(str(extracted))
                data = np.asarray(image.dataobj, dtype=np.float32)
                selection = choose_window(data, args.fraction, leading_discard_fraction=args.leading_discard_fraction)
                start = int(selection["start_zero_based"])
                end = int(selection["end_zero_based"]) + 1
                selected = np.where(np.isfinite(data[:, :, start:end]), np.maximum(data[:, :, start:end], 0), 0).astype(np.float32)
                affine = image.affine.copy()
                affine[:3, 3] += affine[:3, 2] * start
                header = image.header.copy()
                header.set_data_shape(selected.shape)
                output_path = selected_dir / f"{uid}_selected_{selection['selected_slices']}of{selection['n_slices']}.nii.gz"
                nib.save(nib.Nifti1Image(selected, affine, header), str(output_path))

            rows.append({
                "uid": uid,
                "is_pathologic": labels[uid],
                **selection,
                "selected_file": str(output_path),
            })
            if number % 50 == 0 or number == len(uids):
                print(f"completed={number}/{len(uids)}", flush=True)

    rows.sort(key=lambda row: str(row["uid"]))
    with (args.output_dir / "adaptive_slice_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "README.md").write_text(
        "# Adaptive 0.15% hotspot slice selection\n\n"
        f"Processed {len(rows)} raw, unregistered scans. For each scan, voxels above the 99.85th percentile were marked as high-intensity. "
        "The leading 4% of nonempty slices was discarded. The selected contiguous block then maximized the largest 26-connected 3D hotspot component. "
        f"The block length was approximately 7/41 ({args.fraction:.6f}) of the remaining nonempty slices.\n",
        encoding="utf-8",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
