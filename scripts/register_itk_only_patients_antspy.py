#!/usr/bin/env python3
"""Register pathologic scans lacking an ANTs result, using ANTsPy rigid registration."""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import ants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--label", choices=("0.0", "1.0"), default="1.0")
    parser.add_argument("--summary-name", default="registration_summary_ants_patients_itk_only.csv")
    return parser.parse_args()


def source_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members = {}
    for info in archive.infolist():
        name = Path(info.filename).name
        if name.endswith(".nii.gz"):
            members[name.removesuffix(".nii.gz")] = info.filename
    return members


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    transforms_dir = output_dir / "transforms"
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir.mkdir(parents=True, exist_ok=True)

    ants_dir = root / "results/ants_registration_mni"
    itk_dir = root / "results/rigid_registration_mni"
    labels_path = root / "data/external/drive_inputs/train_labels.csv"
    archive_path = root / "data/external/niftis.zip"
    template_path = root / "data/external/drive_inputs/template.nii"
    summary_path = output_dir / args.summary_name

    existing_ants = {p.name.removesuffix(".nii.gz") for p in ants_dir.glob("*.nii.gz")}
    itk_uids = {p.name.removesuffix(".nii.gz") for p in itk_dir.glob("*.nii.gz")}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        labels = {row["uid"]: row["is_pathologic"] for row in csv.DictReader(handle)}
    uids = sorted(uid for uid in itk_uids - existing_ants if labels.get(uid) == args.label)
    if args.limit is not None:
        uids = uids[: args.limit]
    if not uids:
        print("No patient scans requiring ANTs registration.")
        return

    fixed = ants.image_read(str(template_path))
    fieldnames = ["uid", "is_pathologic", "status", "elapsed_sec", "registered_image", "transform", "error"]
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="antspy_itk_only_") as temp_dir, summary_path.open("w", newline="", encoding="utf-8") as handle:
        members = source_members(archive)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, uid in enumerate(uids, start=1):
            started = time.monotonic()
            output_path = output_dir / f"{uid}.nii.gz"
            transform_path = transforms_dir / f"{uid}.mat"
            row = {
                "uid": uid,
                "is_pathologic": labels[uid],
                "status": "error",
                "elapsed_sec": "",
                "registered_image": str(output_path),
                "transform": str(transform_path),
                "error": "",
            }
            try:
                source_name = members[uid]
                source_path = Path(temp_dir) / f"{uid}.nii.gz"
                with archive.open(source_name) as source, source_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                moving = ants.image_read(str(source_path))
                result = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
                ants.image_write(result["warpedmovout"], str(output_path))
                shutil.copyfile(result["fwdtransforms"][0], transform_path)
                row["status"] = "ok"
            except Exception as exc:  # Keep the audit complete if one scan fails.
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_sec"] = f"{time.monotonic() - started:.3f}"
            writer.writerow(row)
            handle.flush()
            print(f"[{index}/{len(uids)}] {uid} {row['status']} {row['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
