#!/usr/bin/env python3
"""Fresh, resumable ANTs rigid registration for the end-to-end five-fold audit."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/niftis_original"
TEMPLATE = ROOT / "submission_ensemble_sigma075_8slice_svm/assets/template.nii"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OUT = ROOT / "results/end_to_end_raw_registration_cv5_seed20260931"
REGISTERED = OUT / "registered"
TRANSFORMS = OUT / "transforms"
WORKERS = min(6, max(1, (os.cpu_count() or 1) - 2))


def register_one(uid: str) -> dict:
    import ants
    started = time.perf_counter()
    output = REGISTERED / f"{uid}.nii.gz"
    transform = TRANSFORMS / f"{uid}.mat"
    if output.exists() and output.stat().st_size > 0 and transform.exists() and transform.stat().st_size > 0:
        return {"uid": uid, "status": "reused", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix=f"e2e_{uid}_") as temporary:
        prefix = str(Path(temporary) / "ants_")
        fixed = ants.image_read(str(TEMPLATE))
        moving = ants.image_read(str(RAW / f"{uid}.nii.gz"))
        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
            outprefix=prefix,
            verbose=False,
        )
        warped_tmp = REGISTERED / f".{uid}.tmp.nii.gz"
        transform_tmp = TRANSFORMS / f".{uid}.tmp.mat"
        ants.image_write(result["warpedmovout"], str(warped_tmp))
        shutil.copy2(result["fwdtransforms"][0], transform_tmp)
        os.replace(warped_tmp, output)
        os.replace(transform_tmp, transform)
    return {"uid": uid, "status": "registered", "seconds": time.perf_counter() - started}


def main() -> None:
    for key in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    REGISTERED.mkdir(parents=True, exist_ok=True)
    TRANSFORMS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open(newline="", encoding="utf-8")))
    uids = sorted(row["uid"] for row in rows)
    missing = [uid for uid in uids if not (RAW / f"{uid}.nii.gz").exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} raw scans; first={missing[:5]}")
    records = []
    wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(register_one, uid): uid for uid in uids}
        for count, future in enumerate(as_completed(futures), 1):
            uid = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                records.append({"uid": uid, "status": "failed", "seconds": 0.0, "error": repr(error)})
            if count % 20 == 0 or count == len(uids):
                failures = sum(row["status"] == "failed" for row in records)
                elapsed = time.perf_counter() - wall_start
                rate = count / max(elapsed, 1e-9)
                print(f"registration={count}/{len(uids)} failures={failures} rate={rate:.3f}_scans_per_sec", flush=True)
                (OUT / "registration_progress.json").write_text(json.dumps({
                    "completed": count, "total": len(uids), "failures": failures,
                    "workers": WORKERS, "elapsed_seconds": elapsed, "records": records,
                }, indent=2) + "\n", encoding="utf-8")
    failed = [row for row in records if row["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Registration failures: {len(failed)}")
    print(f"registration_complete={len(records)} workers={WORKERS}", flush=True)


if __name__ == "__main__":
    main()
