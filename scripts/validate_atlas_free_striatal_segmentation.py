#!/usr/bin/env python3
"""Validate atlas-free caudate/putamen region growing on raw DaT-SPECT scans.

The script intentionally does not use a template, registration, or atlas.  It
creates a stratified visual-validation sample before any SBR classifier is fit.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


ROI_LABELS = {
    "left_caudate": 1,
    "left_putamen": 2,
    "right_caudate": 3,
    "right_putamen": 4,
}
COLORS = {
    "left_caudate": "#00bcd4",
    "left_putamen": "#1976d2",
    "right_caudate": "#ef5350",
    "right_putamen": "#fb8c00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niftis-zip", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--max-physical-radius-mm", type=float, default=25.0)
    parser.add_argument("--min-seed-separation-mm", type=float, default=12.0)
    parser.add_argument("--recent-ring-ratio", type=float, default=0.85)
    parser.add_argument("--putamen-z-offsets", default="-1,-2")
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["uid"]: int(float(row["is_pathologic"])) for row in csv.DictReader(handle)}


def brain_mask(data: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(data), data, 0.0)
    positive = finite[finite > 0]
    if positive.size == 0:
        raise ValueError("scan has no positive intensity voxels")
    # This only discards air/background. It is not an anatomical atlas mask.
    return finite > max(float(np.percentile(positive, 1.0)), float(positive.max()) * 1e-4)


def bounding_box(mask: np.ndarray) -> tuple[slice, slice, slice]:
    points = np.argwhere(mask)
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))  # type: ignore[return-value]


def find_midline(data: np.ndarray, brain: np.ndarray, bounds: tuple[slice, slice, slice]) -> int:
    x0, x1 = bounds[0].start, bounds[0].stop
    centre = (x0 + x1) // 2
    search = range(max(x0 + 4, centre - 10), min(x1 - 4, centre + 11))
    best_plane, best_score = centre, -np.inf
    for plane in search:
        width = min(plane - x0, x1 - plane)
        if width < 4:
            continue
        left = data[plane - width : plane, bounds[1], bounds[2]][::-1]
        right = data[plane : plane + width, bounds[1], bounds[2]]
        support = brain[plane - width : plane, bounds[1], bounds[2]][::-1] & brain[plane : plane + width, bounds[1], bounds[2]]
        a, b = left[support], right[support]
        if a.size < 100 or np.std(a) == 0 or np.std(b) == 0:
            continue
        score = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(score) and score > best_score:
            best_plane, best_score = plane, score
    return best_plane


def central_seed_zone(brain: np.ndarray, bounds: tuple[slice, slice, slice], midline: int, side: str) -> np.ndarray:
    """Central, medial search zone; keeps seed selection away from skull/artifact."""
    x0, x1 = bounds[0].start, bounds[0].stop
    y0, y1 = bounds[1].start, bounds[1].stop
    z0, z1 = bounds[2].start, bounds[2].stop
    zone = np.zeros_like(brain, dtype=bool)
    # The striatum is central.  The medial 80% of each hemisphere is searched.
    if side == "left":
        hx0, hx1 = x0, midline
        xs = slice(max(hx0, int(hx1 - 0.80 * (hx1 - hx0))), max(hx0 + 1, int(hx1 - 0.05 * (hx1 - hx0))))
    else:
        hx0, hx1 = midline, x1
        xs = slice(min(hx1 - 1, int(hx0 + 0.05 * (hx1 - hx0))), min(hx1, int(hx0 + 0.80 * (hx1 - hx0))))
    ys = slice(int(y0 + 0.25 * (y1 - y0)), int(y0 + 0.75 * (y1 - y0)))
    zs = slice(int(z0 + 0.30 * (z1 - z0)), int(z0 + 0.75 * (z1 - z0)))
    zone[xs, ys, zs] = True
    return zone & brain


def highest_voxel(data: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    if not mask.any():
        raise ValueError("empty seed search zone")
    index = np.flatnonzero(mask)[np.argmax(data[mask])]
    return tuple(int(v) for v in np.unravel_index(index, data.shape))


def caudate_seed_zone(zone: np.ndarray, side: str, z_offsets: tuple[int, ...]) -> np.ndarray:
    """Medial central zone with tissue available at an allowed putamen offset."""
    result = zone.copy()
    x_indices = np.flatnonzero(zone.any(axis=(1, 2)))
    if not x_indices.size:
        return result
    x_centre = (int(x_indices.min()) + int(x_indices.max())) / 2.0
    xx = np.arange(zone.shape[0])[:, None, None]
    result &= xx >= x_centre if side == "left" else xx <= x_centre
    valid_z = np.zeros(zone.shape[2], dtype=bool)
    for z in range(zone.shape[2]):
        valid_z[z] = any(0 <= z + offset < zone.shape[2] and zone[:, :, z + offset].any() for offset in z_offsets)
    return result & valid_z[None, None, :]


def putamen_seed(
    data: np.ndarray, zone: np.ndarray, caudate_seed: tuple[int, int, int], side: str,
    spacing: np.ndarray, min_separation_mm: float, z_offsets: tuple[int, ...],
) -> tuple[int, int, int]:
    """Find a lateral local maximum at a requested axial offset from caudate."""
    x, _, z = caudate_seed
    candidate = np.zeros_like(zone, dtype=bool)
    for offset in z_offsets:
        next_z = z + offset
        if 0 <= next_z < zone.shape[2]:
            candidate[:, :, next_z] = zone[:, :, next_z]
    # The putamen is lateral to caudate: away from the midline on each side.
    xx, yy, zz = np.ogrid[: zone.shape[0], : zone.shape[1], : zone.shape[2]]
    candidate &= (xx < x) if side == "left" else (xx > x)
    distance_sq = ((xx - x) * spacing[0]) ** 2 + ((zz - z) * spacing[2]) ** 2
    candidate &= distance_sq >= min_separation_mm ** 2
    return highest_voxel(data, candidate)


def select_seed_pair(
    data: np.ndarray, zone: np.ndarray, side: str, spacing: np.ndarray, min_separation_mm: float,
    z_offsets: tuple[int, ...],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    candidates = caudate_seed_zone(zone, side, z_offsets)
    while candidates.any():
        caudate = highest_voxel(data, candidates)
        try:
            return caudate, putamen_seed(data, zone, caudate, side, spacing, min_separation_mm, z_offsets)
        except ValueError:
            candidates[caudate] = False
    raise ValueError(f"No anatomically separated caudate/putamen seed pair for {side} hemisphere")


def sphere_shell(seed: tuple[int, int, int], radius_mm: float, spacing: np.ndarray, shape: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    radii = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    x0, y0, z0 = seed
    outer = set()
    inner_mm = max(0.0, radius_mm - float(spacing.min()))
    for x in range(max(0, x0 - radii[0]), min(shape[0], x0 + radii[0] + 1)):
        for y in range(max(0, y0 - radii[1]), min(shape[1], y0 + radii[1] + 1)):
            for z in range(max(0, z0 - radii[2]), min(shape[2], z0 + radii[2] + 1)):
                distance = float(np.linalg.norm((np.asarray((x, y, z)) - seed) * spacing))
                if inner_mm < distance <= radius_mm:
                    outer.add((x, y, z))
    return outer


def neighbours26(point: tuple[int, int, int], shape: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    x0, y0, z0 = point
    return {
        (x, y, z)
        for x in range(max(0, x0 - 1), min(shape[0], x0 + 2))
        for y in range(max(0, y0 - 1), min(shape[1], y0 + 2))
        for z in range(max(0, z0 - 1), min(shape[2], z0 + 2))
        if (x, y, z) != point
    }


def grow_pair(
    data: np.ndarray,
    brain: np.ndarray,
    caudate_seed: tuple[int, int, int],
    putamen_seed: tuple[int, int, int],
    spacing: np.ndarray,
    threshold: float,
    max_physical_radius_mm: float,
    recent_ring_ratio: float,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """Lockstep spherical caudate and 26-connected putamen expansion."""
    caudate, putamen = {caudate_seed}, {putamen_seed}
    putamen_frontier = {putamen_seed}
    caudate_active = putamen_active = True
    caudate_previous_ring_mean = float(data[caudate_seed])
    putamen_previous_ring_mean = float(data[putamen_seed])
    radius_mm = float(spacing.min())
    while caudate_active or putamen_active:
        caudate_candidates: set[tuple[int, int, int]] = set()
        if caudate_active and radius_mm <= max_physical_radius_mm:
            caudate_candidates = {
                point for point in sphere_shell(caudate_seed, radius_mm, spacing, data.shape)
                if brain[point] and point not in caudate
            }
        putamen_candidates: set[tuple[int, int, int]] = set()
        if putamen_active:
            for point in putamen_frontier:
                putamen_candidates.update(neighbours26(point, data.shape))
            putamen_candidates = {point for point in putamen_candidates if brain[point] and point not in putamen}

        # Only already-claimed voxels cause a collision.  A shared proposed voxel is
        # assigned to caudate deterministically, so the masks never overlap.
        caudate_hit = bool(caudate_candidates & putamen)
        putamen_hit = bool(putamen_candidates & caudate)
        shared = caudate_candidates & putamen_candidates
        putamen_candidates -= shared
        if caudate_active:
            caudate_mean = float(np.mean([data[point] for point in caudate]))
            candidate_mean = float(np.mean([data[point] for point in caudate_candidates])) if caudate_candidates else 0.0
            if not caudate_candidates or caudate_hit or candidate_mean < threshold * caudate_mean or candidate_mean < recent_ring_ratio * caudate_previous_ring_mean:
                caudate_active = False
            else:
                caudate.update(caudate_candidates)
                caudate_previous_ring_mean = candidate_mean
                radius_mm += float(spacing.min())
        if putamen_active:
            putamen_mean = float(np.mean([data[point] for point in putamen]))
            candidate_mean = float(np.mean([data[point] for point in putamen_candidates])) if putamen_candidates else 0.0
            if not putamen_candidates or putamen_hit or candidate_mean < threshold * putamen_mean or candidate_mean < recent_ring_ratio * putamen_previous_ring_mean:
                putamen_active = False
            else:
                putamen.update(putamen_candidates)
                putamen_previous_ring_mean = candidate_mean
                putamen_frontier = putamen_candidates
    return caudate, putamen


def as_mask(shape: tuple[int, int, int], points: set[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if points:
        mask[tuple(np.asarray(list(points)).T)] = True
    return mask


def save_figure(uid: str, data: np.ndarray, masks: dict[str, np.ndarray], seeds: dict[str, tuple[int, int, int]], path: Path) -> None:
    active_slices = np.unique(np.concatenate([np.flatnonzero(mask.any(axis=(0, 1))) for mask in masks.values()]))
    if not active_slices.size:
        active_slices = np.asarray([data.shape[2] // 2])
    chosen = np.unique(np.round(np.linspace(active_slices.min(), active_slices.max(), min(6, active_slices.size))).astype(int))
    figure, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen), 3.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = float(np.percentile(data[data > 0], 99.5))
    for axis, z in zip(axes, chosen):
        axis.imshow(data[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        for name, mask in masks.items():
            if mask[:, :, z].any():
                axis.contour(mask[:, :, z].T, levels=[0.5], colors=COLORS[name], linewidths=1.5, origin="lower")
            seed = seeds[name]
            if seed[2] == z:
                axis.plot(seed[0], seed[1], marker="x", color=COLORS[name], markersize=6, markeredgewidth=1.5)
        axis.set_title(f"z={z}")
        axis.set_axis_off()
    figure.suptitle(f"{uid}: cyan L-caudate, blue L-putamen, red R-caudate, orange R-putamen", fontsize=10)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def sample_uids(labels: dict[str, int], available: set[str], sample_size: int, seed: int) -> list[str]:
    uids = sorted(set(labels) & available)
    values = np.asarray([labels[uid] for uid in uids])
    if sample_size > len(uids):
        raise ValueError(f"sample_size={sample_size} exceeds {len(uids)} available scans")
    _, selected = train_test_split(np.arange(len(uids)), test_size=sample_size, stratify=values, random_state=seed)
    return [uids[index] for index in sorted(selected)]


def main() -> None:
    args = parse_args()
    z_offsets = tuple(int(value.strip()) for value in args.putamen_z_offsets.split(",") if value.strip())
    if not z_offsets or any(value == 0 for value in z_offsets):
        raise ValueError("putamen z offsets must be nonzero integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    mask_dir = args.output_dir / "masks"
    figure_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    labels = load_labels(args.labels)
    with zipfile.ZipFile(args.niftis_zip) as archive:
        available = {Path(name).stem.removesuffix(".nii") for name in archive.namelist() if name.endswith(".nii.gz")}
        uids = sample_uids(labels, available, args.sample_size, args.seed)
        rows: list[dict[str, object]] = []
        for number, uid in enumerate(uids, start=1):
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted = Path(archive.extract(f"{uid}.nii.gz", temp_dir))
                image = nib.load(str(extracted))
                data = np.asarray(image.dataobj, dtype=np.float32)
            data = np.where(np.isfinite(data), np.maximum(data, 0), 0)
            brain = brain_mask(data)
            bounds = bounding_box(brain)
            midline = find_midline(data, brain, bounds)
            masks: dict[str, np.ndarray] = {}
            seeds: dict[str, tuple[int, int, int]] = {}
            for side in ("left", "right"):
                zone = central_seed_zone(brain, bounds, midline, side)
                caudate_seed, putamen_seed_point = select_seed_pair(
                    data, zone, side, np.asarray(image.header.get_zooms()[:3]), args.min_seed_separation_mm, z_offsets
                )
                caudate, putamen = grow_pair(
                    data, brain, caudate_seed, putamen_seed_point, np.asarray(image.header.get_zooms()[:3]),
                    args.threshold, args.max_physical_radius_mm, args.recent_ring_ratio,
                )
                for structure, points, seed_point in (("caudate", caudate, caudate_seed), ("putamen", putamen, putamen_seed_point)):
                    name = f"{side}_{structure}"
                    masks[name] = as_mask(data.shape, points)
                    seeds[name] = seed_point
            encoded = np.zeros(data.shape, dtype=np.uint8)
            for name, label in ROI_LABELS.items():
                encoded[masks[name]] = label
            nib.save(nib.Nifti1Image(encoded, image.affine, image.header), str(mask_dir / f"{uid}_atlas_free_rois.nii.gz"))
            save_figure(uid, data, masks, seeds, figure_dir / f"{uid}_atlas_free_rois.png")
            rows.append({
                "uid": uid,
                "is_pathologic": labels[uid],
                "midline_x": midline,
                "putamen_z_offsets": ",".join(map(str, z_offsets)),
                "spacing_mm": "x".join(f"{value:.3f}" for value in image.header.get_zooms()[:3]),
                **{f"{name}_seed": ",".join(map(str, seeds[name])) for name in ROI_LABELS},
                **{f"{name}_voxels": int(masks[name].sum()) for name in ROI_LABELS},
                "mask_file": str(mask_dir / f"{uid}_atlas_free_rois.nii.gz"),
                "figure_file": str(figure_dir / f"{uid}_atlas_free_rois.png"),
            })
            print(f"[{number}/{len(uids)}] {uid}", flush=True)
    with (args.output_dir / "validation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "README.md").write_text(
        "# Atlas-free striatal segmentation validation\n\n"
        "This is a raw-volume, no-registration and no-atlas visual-validation run. "
        "Each figure overlays the four region-grown ROIs.  Masks use labels: "
        "1 left caudate, 2 left putamen, 3 right caudate, 4 right putamen.\n\n"
        f"- Stratified sample: {len(rows)} scans\n"
        f"- Running-mean threshold: {args.threshold:.0%}\n"
        f"- 3D putamen connectivity: 26 neighbours\n"
        f"- Emergency caudate radius cap: {args.max_physical_radius_mm:.1f} mm\n",
        encoding="utf-8",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
