#!/usr/bin/env python3
"""Fuse 3D model scores with numeric pre-registration NIfTI header features."""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from train_registered_consecutive_densenet121_3d import DenseNet121_3D
from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


def read_header_features(archive: ZipFile, uid: str) -> dict[str, float]:
    with archive.open(f"{uid}.nii.gz") as compressed:
        with gzip.GzipFile(fileobj=compressed) as nifti_stream:
            header = nib.Nifti1Header.from_fileobj(io.BytesIO(nifti_stream.read(348)))
    dims = np.asarray(header["dim"], dtype=float)
    zooms = np.asarray(header["pixdim"], dtype=float)
    features: dict[str, float] = {
        "raw_dim_x": dims[1], "raw_dim_y": dims[2], "raw_dim_z": dims[3],
        "raw_zoom_x": zooms[1], "raw_zoom_y": zooms[2], "raw_zoom_z": zooms[3],
        "raw_qform_code": float(header["qform_code"]), "raw_sform_code": float(header["sform_code"]),
        "raw_scl_slope": float(header["scl_slope"]), "raw_scl_inter": float(header["scl_inter"]),
        "raw_cal_min": float(header["cal_min"]), "raw_cal_max": float(header["cal_max"]),
        "raw_qfac": float(zooms[0]), "raw_datatype": float(header["datatype"]),
        "raw_bitpix": float(header["bitpix"]), "raw_xyzt_units": float(header["xyzt_units"]),
    }
    affine = header.get_qform()
    if not np.isfinite(affine).all():
        affine = header.get_sform()
    for row in range(3):
        for col in range(4):
            features[f"raw_affine_{row}_{col}"] = float(affine[row, col])
    return features


def model_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device, non_blocking=True))
            output.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(output, dtype=float)


def score_metrics(name: str, truth: np.ndarray, scores: np.ndarray) -> dict[str, float | int | str]:
    prediction = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "model": name, "n_test": int(len(truth)), "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, prediction)), "auroc": float(roc_auc_score(truth, scores)),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/ants_registered_consecutive_resnet18_cbam"
    archive_path = root / "data/external/niftis.zip"
    output = root / "results/ants_registered_3d_models_raw_header_fusion"
    output.mkdir(parents=True, exist_ok=True)
    split_rows = list(csv.DictReader((source / "split.csv").open(newline="", encoding="utf-8")))
    labels = np.asarray([int(row["is_pathologic"]) for row in split_rows], dtype=int)
    split = np.asarray([row["split"] for row in split_rows])
    uids = [row["uid"] for row in split_rows]
    indices = {part: np.flatnonzero(split == part) for part in ("train", "validation", "test")}

    header_rows = []
    with ZipFile(archive_path) as archive:
        for uid in uids:
            header_rows.append(read_header_features(archive, uid))
    feature_names = list(header_rows[0])
    header_matrix = np.asarray([[row[name] for name in feature_names] for row in header_rows], dtype=np.float64)
    np.save(output / "raw_header_features.npy", header_matrix)
    with (output / "raw_header_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", *feature_names])
        writer.writerows([[uid, *values] for uid, values in zip(uids, header_matrix)])

    inputs = np.load(source / "input_windows.npy").astype(np.float32, copy=False)
    tensor_x = torch.from_numpy(inputs).unsqueeze(1)
    tensor_y = torch.from_numpy(labels)
    loaders = {
        part: DataLoader(TensorDataset(tensor_x[index], tensor_y[index]), batch_size=2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        for part, index in indices.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet18CBAM3D().to(device)
    dense = DenseNet121_3D().to(device)
    resnet.load_state_dict(torch.load(root / "results/ants_registered_consecutive_resnet18_cbam/best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    dense.load_state_dict(torch.load(root / "results/ants_registered_consecutive_densenet121_3d/best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    model_scores_by_part = {}
    for part, loader in loaders.items():
        model_scores_by_part[part] = np.column_stack((model_scores(resnet, loader, device), model_scores(dense, loader, device)))

    candidates = []
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        fusion = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=42)),
        ])
        train_idx, val_idx = indices["train"], indices["validation"]
        train_model_scores = model_scores_by_part["train"]
        val_model_scores = model_scores_by_part["validation"]
        x_train = np.column_stack((train_model_scores, header_matrix[train_idx]))
        x_val = np.column_stack((val_model_scores, header_matrix[val_idx]))
        fusion.fit(x_train, labels[train_idx])
        val_probability = fusion.predict_proba(x_val)[:, 1]
        candidates.append((float(roc_auc_score(labels[val_idx], val_probability)), float(accuracy_score(labels[val_idx], val_probability >= 0.5)), c, fusion))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_val_auc, best_val_accuracy, best_c, fusion = candidates[0]

    test_idx = indices["test"]
    test_probability = fusion.predict_proba(np.column_stack((model_scores_by_part["test"], header_matrix[test_idx])))[:, 1]
    ensemble_probability = 0.402 * model_scores_by_part["test"][:, 0] + 0.598 * model_scores_by_part["test"][:, 1]
    metrics = [
        score_metrics("Validation-selected 3D score ensemble (0.402/0.598)", labels[test_idx], ensemble_probability),
        score_metrics("3D scores + raw pre-registration NIfTI header fusion", labels[test_idx], test_probability),
    ]
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "score_3d_ensemble", "score_header_fusion", "predicted_header_fusion"])
        writer.writeheader()
        for position, index in enumerate(test_idx):
            writer.writerow({"uid": uids[index], "actual": int(labels[index]), "score_3d_ensemble": float(ensemble_probability[position]), "score_header_fusion": float(test_probability[position]), "predicted_header_fusion": int(test_probability[position] >= 0.5)})
    with (output / "c_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["validation_auroc", "validation_accuracy", "C"])
        writer.writerows([[auc, accuracy, c] for auc, accuracy, c, _ in candidates])
    (output / "report.md").write_text(
        "# 3D model scores with pre-registration NIfTI header fusion\n\n"
        "- Image inputs: the same registered 16-slice grayscale volumes used by the 3D ResNet-18 and 3D DenseNet-121 models.\n"
        "- Added features: numeric fields from each scan's original pre-registration NIfTI header, including dimensions, voxel spacing, qform/sform codes, affine geometry, datatype, scaling, and calibration fields.\n"
        "- Fusion: standardized, median-imputed logistic regression using both 3D model probabilities and raw-header features.\n"
        f"- Regularization selection: validation AUROC sweep over C values; selected C=`{best_c}` with validation AUROC `{best_val_auc:.4f}` and accuracy `{best_val_accuracy:.4f}`.\n"
        "- Test evaluation: shared untouched 409-scan test set; no test-set weight tuning.\n\n"
        "## Held-out test\n\n"
        + "\n".join(f"- **{row['model']}**: accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}, AUROC={row['auroc']:.4f}" for row in metrics)
        + "\n", encoding="utf-8",
    )
    for row in metrics:
        print(f"{row['model']}: accuracy={row['accuracy']:.4f}, auroc={row['auroc']:.4f}")
    print(f"selected_C={best_c}, validation_auroc={best_val_auc:.4f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
