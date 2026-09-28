#!/usr/bin/env python3
"""Cross-fitted blending of real ResNet and morphometric Gaussian-SVM OOF predictions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from ensemble_3d_tabular_predictions import (
    ProbabilityCalibrator,
    plot_roc_comparison,
    search_blend_weight,
    specificity_at_sensitivity,
)


ROOT = Path(__file__).resolve().parents[1]
RESNET_PATH = ROOT / "results/cv5_real8_post_registration_gaussian/oof_predictions.csv"
SVM_PATH = ROOT / "results/morphometric_unequal_two_band_all1362_cv5/oof_logloss_ranking.csv"
OUT = ROOT / "results/real_resnet16_morph_intensity_svm_ensemble"
CALIBRATIONS = ("none", "platt", "isotonic")
SEED = 20260931


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Return discrimination, calibration, and threshold-0.5 metrics."""
    operating_point = specificity_at_sensitivity(labels, probabilities, 0.95)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "incorrect": int(np.sum((probabilities >= 0.5) != labels)),
        "specificity_at_95_sensitivity": asdict(operating_point),
    }


def load_aligned_predictions() -> pd.DataFrame:
    """Load, align, and validate the two real OOF prediction tables by UID."""
    resnet = pd.read_csv(RESNET_PATH, dtype={"uid": str})[
        ["uid", "actual", "fold", "resnet"]
    ]
    svm = pd.read_csv(SVM_PATH, dtype={"uid": str})[
        ["uid", "actual", "oof_probability"]
    ].rename(columns={"actual": "svm_actual", "oof_probability": "svm"})
    frame = resnet.merge(svm, on="uid", how="inner", validate="one_to_one")
    if len(frame) != len(resnet) or len(frame) != len(svm):
        raise RuntimeError("The ResNet and SVM UID sets do not match exactly.")
    if not np.array_equal(frame["actual"].to_numpy(), frame["svm_actual"].to_numpy()):
        raise RuntimeError("Target-label mismatch between ResNet and SVM prediction files.")
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected matching OOF fold identifiers 1 through 5.")
    return frame.drop(columns="svm_actual").sort_values("uid").reset_index(drop=True)


def cross_fitted_blend(frame: pd.DataFrame, calibration: str) -> tuple[dict, pd.DataFrame]:
    """Tune calibration and weights on four folds, then predict the untouched fifth."""
    labels = frame["actual"].to_numpy(dtype=np.int64)
    folds = frame["fold"].to_numpy(dtype=np.int64)
    raw_3d = frame["resnet"].to_numpy(dtype=np.float64)
    raw_svm = frame["svm"].to_numpy(dtype=np.float64)
    predictions = {
        "resnet": np.full(len(frame), np.nan),
        "svm": np.full(len(frame), np.nan),
        "probability_blend": np.full(len(frame), np.nan),
        "rank_blend": np.full(len(frame), np.nan),
    }
    fold_records = []

    for fold in range(1, 6):
        train = folds != fold
        test = folds == fold
        calibration_3d = ProbabilityCalibrator(calibration, SEED + fold).fit(
            raw_3d[train], labels[train]
        )
        calibration_svm = ProbabilityCalibrator(calibration, SEED + 100 + fold).fit(
            raw_svm[train], labels[train]
        )
        train_3d = calibration_3d.transform(raw_3d[train])
        train_svm = calibration_svm.transform(raw_svm[train])
        test_3d = calibration_3d.transform(raw_3d[test])
        test_svm = calibration_svm.transform(raw_svm[test])
        probability = search_blend_weight(
            labels[train], train_3d, train_svm, test_3d, test_svm, step=0.01
        )
        rank = search_blend_weight(
            labels[train], train_3d, train_svm, test_3d, test_svm,
            rank_average=True, step=0.01,
        )
        predictions["resnet"][test] = test_3d
        predictions["svm"][test] = test_svm
        predictions["probability_blend"][test] = probability.test_predictions
        predictions["rank_blend"][test] = rank.test_predictions
        fold_records.append({
            "fold": fold,
            "meta_train_scans": int(train.sum()),
            "meta_test_scans": int(test.sum()),
            "probability_weight_3d": probability.weight_3d,
            "probability_training_auroc": probability.validation_auc,
            "rank_weight_3d": rank.weight_3d,
            "rank_training_auroc": rank.validation_auc,
            "held_out": {
                name: classification_metrics(labels[test], values[test])
                for name, values in predictions.items()
            },
        })
        print(json.dumps(fold_records[-1]), flush=True)

    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("Cross-fitted prediction arrays contain missing values.")
    pooled = {
        name: classification_metrics(labels, values)
        for name, values in predictions.items()
    }
    result = {
        "calibration": calibration,
        "evaluation": "five-fold cross-fitted meta-ensemble over aligned base-model OOF predictions",
        "n_scans": int(len(frame)),
        "pooled": pooled,
        "mean_probability_weight_3d": float(np.mean([
            row["probability_weight_3d"] for row in fold_records
        ])),
        "mean_rank_weight_3d": float(np.mean([
            row["rank_weight_3d"] for row in fold_records
        ])),
        "folds": fold_records,
    }
    output = frame[["uid", "actual", "fold"]].copy()
    for name, values in predictions.items():
        output[name] = values
    return result, output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_aligned_predictions()
    all_results = {}
    for calibration in CALIBRATIONS:
        print(f"\ncalibration={calibration}", flush=True)
        result, predictions = cross_fitted_blend(frame, calibration)
        all_results[calibration] = result
        predictions.to_csv(OUT / f"oof_predictions_{calibration}.csv", index=False)
        plot_roc_comparison(
            predictions["actual"].to_numpy(),
            {
                "ResNet-18 (16 epochs)": predictions["resnet"].to_numpy(),
                "Morphometric + intensity RBF SVM": predictions["svm"].to_numpy(),
                "Weighted probability blend": predictions["probability_blend"].to_numpy(),
                "Rank blend": predictions["rank_blend"].to_numpy(),
            },
            OUT / f"roc_{calibration}.png",
        )
        print(f"pooled={json.dumps(result['pooled'], indent=2)}", flush=True)

    best_probability = max(
        CALIBRATIONS,
        key=lambda name: all_results[name]["pooled"]["probability_blend"]["auroc"],
    )
    best_rank = max(
        CALIBRATIONS,
        key=lambda name: all_results[name]["pooled"]["rank_blend"]["auroc"],
    )
    summary = {
        "resnet_source": str(RESNET_PATH),
        "svm_source": str(SVM_PATH),
        "external_test_set_available": False,
        "methodological_note": (
            "Calibration and blend weights were fit on four aligned OOF folds and "
            "applied to the untouched fifth fold. No external shared test predictions existed."
        ),
        "best_probability_calibration_by_auroc": best_probability,
        "best_rank_calibration_by_auroc": best_rank,
        "results": all_results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nsummary={OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
