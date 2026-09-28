#!/usr/bin/env python3
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "external" / "niftis_original"
OUTPUT = ROOT / "results" / "acquisition_style_audit"
N_CLUSTERS = 10
SEED = 20260931


def scalar_description(value: bytes) -> float:
    match = re.search(r"Time=([0-9.]+)", bytes(value).decode("utf-8", errors="ignore"))
    return float(match.group(1)) if match else np.nan


def extract_features(path: Path) -> dict[str, float | str]:
    image = nib.load(path)
    header = image.header
    data = np.asarray(image.dataobj, dtype=np.float32)
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    if positive.size == 0:
        raise RuntimeError(f"No positive voxels: {path.name}")
    stride = max(1, positive.size // 100_000)
    percentile_sample = positive[::stride]
    com = center_of_mass(np.where(np.isfinite(data), np.maximum(data, 0), 0))
    shape = data.shape
    affine = image.affine
    return {
        "uid": path.name.removesuffix(".nii.gz"),
        "dim_x": shape[0], "dim_y": shape[1], "dim_z": shape[2],
        "spacing_x": float(header["pixdim"][1]), "spacing_y": float(header["pixdim"][2]), "spacing_z": float(header["pixdim"][3]),
        "datatype": int(header["datatype"]), "bitpix": int(header["bitpix"]),
        "qform_code": int(header["qform_code"]), "sform_code": int(header["sform_code"]),
        "time_description": scalar_description(header["descrip"]),
        "positive_fraction": float(positive.size / finite.size),
        "positive_mean": float(positive.mean()), "positive_std": float(positive.std()),
        "positive_p01": float(np.percentile(percentile_sample, 1)), "positive_p50": float(np.percentile(percentile_sample, 50)),
        "positive_p99": float(np.percentile(percentile_sample, 99)),
        "com_x_fraction": float(com[0] / max(shape[0] - 1, 1)),
        "com_y_fraction": float(com[1] / max(shape[1] - 1, 1)),
        "com_z_fraction": float(com[2] / max(shape[2] - 1, 1)),
        "affine_det": float(np.linalg.det(affine[:3, :3])),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT.glob("*.nii.gz"))
    rows = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for index, row in enumerate(executor.map(extract_features, paths), start=1):
            rows.append(row)
            if index % 100 == 0 or index == len(paths):
                print(f"Extracted {index}/{len(paths)} scan feature rows", flush=True)
        if index % 100 == 0 or index == len(paths):
            print(f"Extracted {index}/{len(paths)} scan feature rows", flush=True)
    features = pd.DataFrame(rows)
    numeric_columns = [column for column in features.columns if column != "uid"]
    matrix = features[numeric_columns].copy()
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    variable_columns = [column for column in numeric_columns if matrix[column].nunique() > 1]
    scaled = StandardScaler().fit_transform(matrix[variable_columns])
    features["acquisition_style_cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=50).fit_predict(scaled)
    features.to_csv(OUTPUT / "scan_features.csv", index=False)
    summary = features.groupby("acquisition_style_cluster")[variable_columns].agg(["count", "mean", "std"])
    summary.to_csv(OUTPUT / "cluster_summary.csv")
    pd.DataFrame({"feature": numeric_columns, "unique_values": [int(features[column].nunique()) for column in numeric_columns]}).to_csv(OUTPUT / "feature_cardinality.csv", index=False)
    print(features.groupby("acquisition_style_cluster").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
