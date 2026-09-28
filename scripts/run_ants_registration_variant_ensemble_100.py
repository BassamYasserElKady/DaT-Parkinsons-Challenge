#!/usr/bin/env python3
"""Compare 15 ANTs rigid/affine variants on 100 held-out scans."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import shutil
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import ants
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D, choose_window


VARIANTS = [
    {"name": "rigid_mattes_standard", "type": "Rigid", "metric": "mattes", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "rigid_mattes_fast", "type": "Rigid", "metric": "mattes", "iterations": (40, 20, 0), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "rigid_mattes_deep", "type": "Rigid", "metric": "mattes", "iterations": (200, 100, 50, 20), "shrink": (8, 4, 2, 1), "smooth": (3, 2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "rigid_mattes_fine_step", "type": "Rigid", "metric": "mattes", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.05, "sampling": 64, "random": 0.20},
    {"name": "rigid_gc_standard", "type": "Rigid", "metric": "gc", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "rigid_meansquares_standard", "type": "Rigid", "metric": "meansquares", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "affine_mattes_standard", "type": "Affine", "metric": "mattes", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "affine_mattes_fast", "type": "Affine", "metric": "mattes", "iterations": (40, 20, 0), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "affine_mattes_deep", "type": "Affine", "metric": "mattes", "iterations": (200, 100, 50, 20), "shrink": (8, 4, 2, 1), "smooth": (3, 2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "affine_mattes_fine_step", "type": "Affine", "metric": "mattes", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.05, "sampling": 64, "random": 0.20},
    {"name": "affine_mattes_dense_sampling", "type": "Affine", "metric": "mattes", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 128, "random": 0.50},
    {"name": "affine_gc_standard", "type": "Affine", "metric": "gc", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "affine_gc_deep", "type": "Affine", "metric": "gc", "iterations": (200, 100, 50, 20), "shrink": (8, 4, 2, 1), "smooth": (3, 2, 1, 0), "grad": 0.1, "sampling": 64, "random": 0.30},
    {"name": "affine_meansquares_standard", "type": "Affine", "metric": "meansquares", "iterations": (100, 70, 50), "shrink": (4, 2, 1), "smooth": (2, 1, 0), "grad": 0.1, "sampling": 32, "random": 0.20},
    {"name": "affine_meansquares_deep", "type": "Affine", "metric": "meansquares", "iterations": (200, 100, 50, 20), "shrink": (8, 4, 2, 1), "smooth": (3, 2, 1, 0), "grad": 0.1, "sampling": 64, "random": 0.30},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n-scans", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def header_features(archive: ZipFile, uid: str) -> dict[str, float]:
    with archive.open(f"{uid}.nii.gz") as compressed:
        with gzip.GzipFile(fileobj=compressed) as stream:
            header = nib.Nifti1Header.from_fileobj(io.BytesIO(stream.read(348)))
    dims = np.asarray(header["dim"], dtype=float)
    zooms = np.asarray(header["pixdim"], dtype=float)
    affine = header.get_qform()
    if not np.isfinite(affine).all():
        affine = header.get_sform()
    result = {
        "raw_dim_x": dims[1], "raw_dim_y": dims[2], "raw_dim_z": dims[3],
        "raw_zoom_x": zooms[1], "raw_zoom_y": zooms[2], "raw_zoom_z": zooms[3],
        "raw_qform_code": float(header["qform_code"]), "raw_sform_code": float(header["sform_code"]),
        "raw_datatype": float(header["datatype"]), "raw_bitpix": float(header["bitpix"]),
    }
    for row in range(3):
        for col in range(4):
            result[f"raw_affine_{row}_{col}"] = float(affine[row, col])
    return result


def preprocess(path: Path) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    selection = choose_window(data)
    start = int(selection["start_zero_based"])
    end = int(selection["end_zero_based"]) + 1
    volume = np.where(np.isfinite(data[:, :, start:end]), np.maximum(data[:, :, start:end], 0), 0).astype(np.float32)
    volume = np.transpose(volume, (2, 1, 0))
    positive = volume[volume > 0]
    if positive.size:
        low, high = np.percentile(positive, [1, 99])
        if high <= low:
            high = low + 1.0
        volume = np.clip((volume - low) / (high - low), 0.0, 1.0)
    return ndimage.zoom(volume, (16 / volume.shape[0], 64 / volume.shape[1], 64 / volume.shape[2]), order=1).astype(np.float32)


def model_probability(models: tuple[torch.nn.Module, torch.nn.Module], volume: np.ndarray, device: torch.device) -> tuple[float, float]:
    batch = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return tuple(float(torch.softmax(model(batch), dim=1)[0, 1].cpu()) for model in models)


def metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    prediction = score >= 0.5
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "n_scans": int(len(truth)), "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, prediction)), "auroc": float(roc_auc_score(truth, score)),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
    }


def main() -> None:
    args = parse_args()
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    root = args.root.resolve()
    output = root / "results/ants_registration_variant_ensemble_100"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "results/ants_registered_consecutive_resnet18_cbam"
    rows = list(csv.DictReader((source / "split.csv").open(newline="", encoding="utf-8")))
    test_rows = [row for row in rows if row["split"] == "test"]
    _ , selected = train_test_split(test_rows, test_size=args.n_scans, stratify=[row["is_pathologic"] for row in test_rows], random_state=args.random_state)
    selected.sort(key=lambda row: row["uid"])
    with ZipFile(root / "data/external/niftis.zip") as archive:
        raw_headers = {row["uid"]: header_features(archive, row["uid"]) for row in selected}
        raw_dir = output / "raw_selected"
        raw_dir.mkdir(exist_ok=True)
        for row in selected:
            destination = raw_dir / f"{row['uid']}.nii.gz"
            if not destination.exists():
                with archive.open(f"{row['uid']}.nii.gz") as source_file, destination.open("wb") as destination_file:
                    shutil.copyfileobj(source_file, destination_file)
    with (output / "selected_scans.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["uid", "is_pathologic", "split", *raw_headers[selected[0]["uid"]]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, **raw_headers[row["uid"]]})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet18CBAM3D().to(device)
    dense = DenseNet121_3D().to(device)
    resnet.load_state_dict(torch.load(root / "results/ants_registered_consecutive_resnet18_cbam/best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    dense.load_state_dict(torch.load(root / "results/ants_registered_consecutive_densenet121_3d/best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    resnet.eval(); dense.eval()
    models = (resnet, dense)
    labels = np.asarray([int(row["is_pathologic"]) for row in selected], dtype=int)
    baseline_input = np.load(source / "input_windows.npy").astype(np.float32, copy=False)
    uid_to_index = {row["uid"]: index for index, row in enumerate(rows)}
    baseline_scores = []
    for row in selected:
        volume = baseline_input[uid_to_index[row["uid"]]]
        baseline_scores.append(model_probability(models, volume, device))
    baseline_scores = np.asarray(baseline_scores)
    summary_rows = [{"variant": "baseline_existing_ants", "registration_family": "baseline", **metrics(labels, 0.402 * baseline_scores[:, 0] + 0.598 * baseline_scores[:, 1])}]

    fixed = ants.image_read(str(root / "data/external/drive_inputs/template.nii"))
    for number, variant in enumerate(VARIANTS, start=1):
        variant_dir = output / "registered" / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        prediction_rows = []
        started_variant = time.monotonic()
        print(f"variant={number}/15 name={variant['name']}", flush=True)
        for position, row in enumerate(selected, start=1):
            uid = row["uid"]
            registered_path = variant_dir / f"{uid}.nii.gz"
            started = time.monotonic()
            try:
                if not registered_path.exists():
                    moving = ants.image_read(str(output / "raw_selected" / f"{uid}.nii.gz"))
                    with tempfile.TemporaryDirectory(prefix="ants_variant_") as temp_dir:
                        registration = ants.registration(
                            fixed=fixed,
                            moving=moving,
                            type_of_transform=variant["type"],
                            outprefix=str(Path(temp_dir) / f"{uid}_"),
                            grad_step=variant["grad"],
                            aff_metric=variant["metric"],
                            aff_sampling=variant["sampling"],
                            aff_random_sampling_rate=variant["random"],
                            aff_iterations=variant["iterations"],
                            aff_shrink_factors=variant["shrink"],
                            aff_smoothing_sigmas=variant["smooth"],
                            verbose=False,
                            singleprecision=True,
                        )
                        ants.image_write(registration["warpedmovout"], str(registered_path))
                volume = preprocess(registered_path)
                resnet_score, dense_score = model_probability(models, volume, device)
                prediction_rows.append({"uid": uid, "actual": int(row["is_pathologic"]), "resnet_score": resnet_score, "densenet_score": dense_score, "ensemble_score": 0.402 * resnet_score + 0.598 * dense_score})
                print(f"  scan={position}/{len(selected)} uid={uid} sec={time.monotonic() - started:.1f}", flush=True)
            except Exception as exc:
                print(f"  scan={position}/{len(selected)} uid={uid} ERROR={type(exc).__name__}: {exc}", flush=True)
        if len(prediction_rows) != len(selected):
            raise RuntimeError(f"variant {variant['name']} produced {len(prediction_rows)}/{len(selected)} predictions")
        with (variant_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader(); writer.writerows(prediction_rows)
        variant_metrics = {"variant": variant["name"], "registration_family": variant["type"].lower(), **metrics(labels, np.asarray([item["ensemble_score"] for item in prediction_rows]))}
        summary_rows.append(variant_metrics)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader(); writer.writerows(summary_rows)
        print(f"completed={variant['name']} elapsed_sec={time.monotonic() - started_variant:.1f} accuracy={variant_metrics['accuracy']:.4f} auroc={variant_metrics['auroc']:.4f}", flush=True)
    print(f"output={output}")


if __name__ == "__main__":
    main()
