#!/usr/bin/env python3
"""Direct same-split comparison: original 22-feature SVM vs MIP6 feature addition."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from holdout80_fixed299_four_model import build, metrics as original_metrics, SEED, LABELS  # noqa: E402

SPLIT = ROOT / "results" / "holdout80_single_fixed299_four_model" / "split.npz"
SHAPES = ROOT / "results" / "adaptive8_mip6_guided_3d_quarter_shapes" / "labels"
MAPS = ROOT / "results" / "adaptive8_mip6_guided_sliding_intensity_pyramid" / "maps"
OUT = ROOT / "results" / "holdout80_compare_original_svm_vs_mip6_features"
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def extra(uid: str) -> np.ndarray:
    label = np.asarray(nib.load(str(SHAPES / f"{uid}_adaptive8_3d_quarters.nii.gz")).dataobj, dtype=np.uint8)
    left = np.asarray([np.sum(label == code) for code in range(1, 5)], float)
    right = np.asarray([np.sum(label == code) for code in range(5, 9)], float)
    features = [left.sum(), right.sum(), *left, *right, *(left / max(left.sum(), 1.0)), *(right / max(right.sum(), 1.0))]
    sparse = np.load(MAPS / f"{uid}_adaptive8_sliding_pyramid.npz")
    raw = sparse["intensity_1"].astype(float); pos = raw[raw > 0]
    scale = max(float(np.median(pos)) if len(pos) else 1.0, 1e-6)
    for side in (0, 1):
        select = sparse["side"] == side
        for key in ("intensity_1", "intensity_2", "intensity_3"):
            values = sparse[key][select].astype(float) / scale
            if len(values): features += [values.mean(), values.std(), *np.quantile(values, QUANTILES)]
            else: features += [0.0] * 7
    return np.asarray(features, dtype=float)


def summary_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    result = original_metrics(y, p)
    return {k: int(v) if isinstance(v, (int, np.integer)) else float(v) for k, v in result.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open()))
    y = np.asarray([int(row["is_pathologic"]) for row in rows])
    split = np.load(SPLIT)
    dev, test = split["development"], split["test"]
    # Exact cached SVM features and exact split used by the earlier 0.9593 run.
    _, _, original22 = build(rows)
    add = np.asarray([extra(row["uid"]) for row in rows], dtype=float)
    combined = np.column_stack((original22, add))
    prediction = {}
    for name, features in (("original22", original22), ("plus_mip6_shape_intensity", combined)):
        model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED + 3))
        model.fit(features[dev], y[dev])
        prediction[name] = model.predict_proba(features[test])[:, 1]
    result = {
        "evaluation": "same saved stratified 80/20 split and same original 22-feature SVM inputs as holdout80_single_fixed299_four_model",
        "development_scans": int(len(dev)), "test_scans": int(len(test)),
        "base_feature_count": int(original22.shape[1]), "added_feature_count": int(add.shape[1]), "combined_feature_count": int(combined.shape[1]),
        "original22": summary_metric(y[test], prediction["original22"]),
        "plus_mip6_shape_intensity": summary_metric(y[test], prediction["plus_mip6_shape_intensity"]),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "test_predictions.npz", uid=np.asarray([rows[i]["uid"] for i in test]), y=y[test], original22=prediction["original22"], plus_mip6_shape_intensity=prediction["plus_mip6_shape_intensity"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
