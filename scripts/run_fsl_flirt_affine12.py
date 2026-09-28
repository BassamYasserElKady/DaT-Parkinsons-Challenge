#!/usr/bin/env python3

import csv
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FSLDIR = ROOT / ".tools" / "fsl-flirt"
FLIRT = FSLDIR / "bin" / "flirt"
INPUT_DIR = ROOT / "data" / "external" / "niftis_original"
TEMPLATE = ROOT / "submission_src_ensemble" / "assets" / "template.nii"
OUTPUT_DIR = ROOT / "results" / "fsl_flirt_affine12"
REGISTERED_DIR = OUTPUT_DIR / "registered"
MATRIX_DIR = OUTPUT_DIR / "matrices"
LOG_DIR = OUTPUT_DIR / "logs"
SUMMARY_PATH = OUTPUT_DIR / "registration_summary.csv"
WORKERS = min(16, os.cpu_count() or 1)


def register(scan_path_str):
    scan_path = Path(scan_path_str)
    uid = scan_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    output_path = REGISTERED_DIR / f"{uid}.nii.gz"
    matrix_path = MATRIX_DIR / f"{uid}.mat"
    log_path = LOG_DIR / f"{uid}.log"

    if output_path.exists() and output_path.stat().st_size > 0 and matrix_path.exists():
        return uid, "existing", 0.0, ""

    env = os.environ.copy()
    env["FSLDIR"] = str(FSLDIR)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PATH"] = f"{FSLDIR / 'bin'}:{env.get('PATH', '')}"

    command = [
        str(FLIRT),
        "-in", str(scan_path),
        "-ref", str(TEMPLATE),
        "-out", str(output_path),
        "-omat", str(matrix_path),
        "-dof", "12",
        "-cost", "corratio",
        "-searchcost", "corratio",
        "-interp", "trilinear",
    ]

    started = time.monotonic()
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    log_path.write_text(
        "COMMAND\n" + " ".join(command) +
        "\n\nSTDOUT\n" + completed.stdout +
        "\n\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        matrix_path.unlink(missing_ok=True)
        return uid, "failed", elapsed, completed.stderr.strip()
    if not output_path.exists() or not matrix_path.exists():
        return uid, "failed", elapsed, "FLIRT returned success without both outputs"
    return uid, "completed", elapsed, ""


def write_summary(rows):
    temporary = SUMMARY_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uid", "status", "seconds", "error"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["uid"]))
    temporary.replace(SUMMARY_PATH)


def main():
    if not FLIRT.exists():
        raise FileNotFoundError(f"FLIRT not found: {FLIRT}")
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    REGISTERED_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    scans = sorted(INPUT_DIR.glob("*.nii.gz")) + sorted(INPUT_DIR.glob("*.nii"))
    print(f"template={TEMPLATE}", flush=True)
    print(f"scans={len(scans)} workers={WORKERS} dof=12 cost=corratio", flush=True)

    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(register, str(path)): path for path in scans}
        for index, future in enumerate(as_completed(futures), start=1):
            uid, status, seconds, error = future.result()
            rows.append({"uid": uid, "status": status, "seconds": f"{seconds:.3f}", "error": error})
            write_summary(rows)
            print(
                f"[{index:04d}/{len(scans):04d}] {uid} {status} "
                f"seconds={seconds:.1f} elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )

    completed = sum(row["status"] in {"completed", "existing"} for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"finished completed={completed} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
