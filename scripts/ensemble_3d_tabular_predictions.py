#!/usr/bin/env python3
"""Blend OOF and held-out predictions from 3D and tabular binary classifiers.

The blend weight and any optional calibration models are learned using only
out-of-fold (OOF) validation predictions. The held-out test labels are used
only for final reporting and never for model selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


CalibrationMethod = Literal["none", "platt", "isotonic"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BlendSearchResult:
    """Result of a one-dimensional blend-weight search."""

    weight_3d: float
    validation_auc: float
    validation_predictions: FloatArray
    test_predictions: FloatArray


@dataclass(frozen=True)
class SpecificityResult:
    """Operating point attaining at least the requested sensitivity."""

    specificity: float
    sensitivity: float
    threshold: float


class ProbabilityCalibrator:
    """Fit and apply identity, Platt, or isotonic probability calibration."""

    def __init__(self, method: CalibrationMethod = "none", seed: int = 42) -> None:
        if method not in {"none", "platt", "isotonic"}:
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self.seed = seed
        self.model: LogisticRegression | IsotonicRegression | None = None

    @staticmethod
    def _logit(probabilities: FloatArray) -> FloatArray:
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, probabilities: ArrayLike, labels: ArrayLike) -> "ProbabilityCalibrator":
        """Fit the requested calibration map on OOF predictions."""
        p = as_probabilities(probabilities, "calibration probabilities")
        y = as_binary_labels(labels, "calibration labels")
        require_equal_length(y, p, "calibration labels", "calibration probabilities")

        if self.method == "platt":
            self.model = LogisticRegression(
                solver="lbfgs", C=1e6, max_iter=10_000, random_state=self.seed
            ).fit(self._logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(p, y)
        return self

    def transform(self, probabilities: ArrayLike) -> FloatArray:
        """Apply the fitted calibration map."""
        p = as_probabilities(probabilities, "probabilities")
        if self.method == "none":
            return p.copy()
        if self.model is None:
            raise RuntimeError("The calibrator must be fitted before transform().")
        if self.method == "platt":
            assert isinstance(self.model, LogisticRegression)
            return self.model.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]
        assert isinstance(self.model, IsotonicRegression)
        return np.asarray(self.model.predict(p), dtype=np.float64)


def as_binary_labels(values: ArrayLike, name: str) -> NDArray[np.int64]:
    """Convert and validate a one-dimensional binary label array."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least 2 values.")
    if not np.all(np.isfinite(array)) or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only finite binary values 0 and 1.")
    result = array.astype(np.int64, copy=False)
    if np.unique(result).size != 2:
        raise ValueError(f"{name} must contain both classes to calculate AUROC.")
    return result


def as_probabilities(values: ArrayLike, name: str) -> FloatArray:
    """Convert and validate a one-dimensional probability array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least 2 values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be bounded within [0, 1].")
    return array


def require_equal_length(a: ArrayLike, b: ArrayLike, a_name: str, b_name: str) -> None:
    """Raise a clear error when paired arrays have different lengths."""
    if len(a) != len(b):
        raise ValueError(f"{a_name} and {b_name} differ in length: {len(a)} != {len(b)}")


def percentile_ranks(probabilities: ArrayLike) -> FloatArray:
    """Map probabilities to normalized average ranks in the closed interval [0, 1]."""
    p = as_probabilities(probabilities, "rank probabilities")
    if p.size == 1:
        return np.full(1, 0.5, dtype=np.float64)
    ranks = rankdata(p, method="average")
    return np.asarray((ranks - 1.0) / (p.size - 1.0), dtype=np.float64)


def search_blend_weight(
    validation_labels: ArrayLike,
    validation_3d: ArrayLike,
    validation_tabular: ArrayLike,
    test_3d: ArrayLike,
    test_tabular: ArrayLike,
    *,
    rank_average: bool = False,
    step: float = 0.01,
) -> BlendSearchResult:
    """Find the 3D-model weight maximizing OOF AUROC and apply it to test data.

    Ties in AUROC are resolved deterministically by choosing the weight closest
    to 0.5, then the smaller weight. This avoids arbitrary extreme weights on
    broad AUROC plateaus.
    """
    y = as_binary_labels(validation_labels, "validation labels")
    val_3d = as_probabilities(validation_3d, "OOF 3D predictions")
    val_tab = as_probabilities(validation_tabular, "OOF tabular predictions")
    tst_3d = as_probabilities(test_3d, "test 3D predictions")
    tst_tab = as_probabilities(test_tabular, "test tabular predictions")
    require_equal_length(y, val_3d, "validation labels", "OOF 3D predictions")
    require_equal_length(y, val_tab, "validation labels", "OOF tabular predictions")
    require_equal_length(tst_3d, tst_tab, "test 3D predictions", "test tabular predictions")
    if not (0.0 < step <= 1.0):
        raise ValueError("step must be in (0, 1].")

    if rank_average:
        val_3d, val_tab = percentile_ranks(val_3d), percentile_ranks(val_tab)
        tst_3d, tst_tab = percentile_ranks(tst_3d), percentile_ranks(tst_tab)

    weights = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    candidates: list[tuple[float, float, FloatArray]] = []
    for weight in weights:
        blended = weight * val_3d + (1.0 - weight) * val_tab
        candidates.append((float(roc_auc_score(y, blended)), float(weight), blended))
    best_auc = max(item[0] for item in candidates)
    tied = [item for item in candidates if np.isclose(item[0], best_auc, atol=1e-12, rtol=0.0)]
    best_auc, best_weight, best_validation = min(
        tied, key=lambda item: (abs(item[1] - 0.5), item[1])
    )
    best_test = best_weight * tst_3d + (1.0 - best_weight) * tst_tab
    return BlendSearchResult(best_weight, best_auc, best_validation, best_test)


def specificity_at_sensitivity(
    labels: ArrayLike, probabilities: ArrayLike, minimum_sensitivity: float = 0.95
) -> SpecificityResult:
    """Return maximum specificity among ROC points meeting a sensitivity floor."""
    y = as_binary_labels(labels, "labels")
    p = as_probabilities(probabilities, "predictions")
    require_equal_length(y, p, "labels", "predictions")
    if not (0.0 < minimum_sensitivity <= 1.0):
        raise ValueError("minimum_sensitivity must be in (0, 1].")
    fpr, tpr, thresholds = roc_curve(y, p)
    eligible = np.flatnonzero(tpr >= minimum_sensitivity)
    if eligible.size == 0:
        raise RuntimeError("No ROC operating point met the sensitivity requirement.")
    best = eligible[np.argmin(fpr[eligible])]
    return SpecificityResult(
        specificity=float(1.0 - fpr[best]),
        sensitivity=float(tpr[best]),
        threshold=float(thresholds[best]),
    )


def plot_roc_comparison(
    labels: ArrayLike,
    predictions: dict[str, ArrayLike],
    output_path: str | Path,
) -> None:
    """Save comparative held-out ROC curves as a PNG."""
    y = as_binary_labels(labels, "plot labels")
    figure, axis = plt.subplots(figsize=(8, 7))
    for name, values in predictions.items():
        p = as_probabilities(values, name)
        require_equal_length(y, p, "plot labels", name)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        axis.plot(fpr, tpr, linewidth=2, label=f"{name} (AUROC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="0.5", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="Sensitivity", title="Held-out ROC comparison")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def evaluate_ensembles(
    y_validation: ArrayLike,
    y_test: ArrayLike,
    oof_preds_3d: ArrayLike,
    oof_preds_tabular: ArrayLike,
    test_preds_3d: ArrayLike,
    test_preds_tabular: ArrayLike,
    *,
    calibration: CalibrationMethod = "none",
    grid_step: float = 0.01,
    roc_output_path: str | Path = "ensemble_roc.png",
    seed: int = 42,
) -> dict[str, object]:
    """Calibrate, tune, evaluate, print, and plot both ensemble methods."""
    y_val = as_binary_labels(y_validation, "validation labels")
    y_tst = as_binary_labels(y_test, "test labels")
    val_3d = as_probabilities(oof_preds_3d, "OOF 3D predictions")
    val_tab = as_probabilities(oof_preds_tabular, "OOF tabular predictions")
    tst_3d = as_probabilities(test_preds_3d, "test 3D predictions")
    tst_tab = as_probabilities(test_preds_tabular, "test tabular predictions")
    require_equal_length(y_val, val_3d, "validation labels", "OOF 3D predictions")
    require_equal_length(y_val, val_tab, "validation labels", "OOF tabular predictions")
    require_equal_length(y_tst, tst_3d, "test labels", "test 3D predictions")
    require_equal_length(y_tst, tst_tab, "test labels", "test tabular predictions")

    calibrator_3d = ProbabilityCalibrator(calibration, seed).fit(val_3d, y_val)
    calibrator_tab = ProbabilityCalibrator(calibration, seed + 1).fit(val_tab, y_val)
    val_3d = calibrator_3d.transform(val_3d)
    val_tab = calibrator_tab.transform(val_tab)
    tst_3d = calibrator_3d.transform(tst_3d)
    tst_tab = calibrator_tab.transform(tst_tab)

    probability_blend = search_blend_weight(
        y_val, val_3d, val_tab, tst_3d, tst_tab, step=grid_step
    )
    rank_blend = search_blend_weight(
        y_val, val_3d, val_tab, tst_3d, tst_tab, rank_average=True, step=grid_step
    )
    test_predictions = {
        "3D ResNet-18": tst_3d,
        "Gaussian RBF SVM": tst_tab,
        "Weighted probability ensemble": probability_blend.test_predictions,
        "Rank ensemble": rank_blend.test_predictions,
    }
    validation_auc = {
        "3D ResNet-18": float(roc_auc_score(y_val, val_3d)),
        "Gaussian RBF SVM": float(roc_auc_score(y_val, val_tab)),
        "Weighted probability ensemble": probability_blend.validation_auc,
        "Rank ensemble": rank_blend.validation_auc,
    }
    test_auc = {
        name: float(roc_auc_score(y_tst, predictions))
        for name, predictions in test_predictions.items()
    }
    operating_points = {
        name: specificity_at_sensitivity(y_tst, predictions, 0.95)
        for name, predictions in test_predictions.items()
    }

    print(f"Calibration: {calibration}")
    print("\nStandalone AUROCs")
    print(f"  3D validation:       {validation_auc['3D ResNet-18']:.4f}")
    print(f"  3D held-out test:    {test_auc['3D ResNet-18']:.4f}")
    print(f"  Tabular validation:  {validation_auc['Gaussian RBF SVM']:.4f}")
    print(f"  Tabular held-out:    {test_auc['Gaussian RBF SVM']:.4f}")
    print("\nWeighted probability average")
    print(f"  Optimal 3D weight:   {probability_blend.weight_3d:.2f}")
    print(f"  Validation AUROC:    {probability_blend.validation_auc:.4f}")
    print(f"  Held-out test AUROC: {test_auc['Weighted probability ensemble']:.4f}")
    print("\nRank average")
    print(f"  Optimal 3D weight:   {rank_blend.weight_3d:.2f}")
    print(f"  Validation AUROC:    {rank_blend.validation_auc:.4f}")
    print(f"  Held-out test AUROC: {test_auc['Rank ensemble']:.4f}")
    print("\nHeld-out specificity at >=95% sensitivity")
    for name, point in operating_points.items():
        print(
            f"  {name:31s} specificity={point.specificity:.4f} "
            f"sensitivity={point.sensitivity:.4f} threshold={point.threshold:.6f}"
        )

    plot_roc_comparison(y_tst, test_predictions, roc_output_path)
    print(f"\nROC plot saved to: {Path(roc_output_path).resolve()}")
    return {
        "calibration": calibration,
        "validation_auc": validation_auc,
        "test_auc": test_auc,
        "probability_blend": probability_blend,
        "rank_blend": rank_blend,
        "specificity_at_95_sensitivity": operating_points,
    }


def synthetic_predictions(n: int, seed: int) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    """Generate correlated model predictions with realistic high discrimination."""
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.55, size=n).astype(np.int64)
    direction = 2.0 * labels - 1.0
    shared_noise = rng.normal(0.0, 0.65, size=n)
    logits_3d = 1.15 * direction + shared_noise + rng.normal(0.0, 0.72, size=n)
    logits_tabular = 1.08 * direction + 0.72 * shared_noise + rng.normal(0.0, 0.82, size=n)
    return labels, expit(logits_3d), expit(logits_tabular)


def main() -> None:
    """Run a reproducible N=500 synthetic end-to-end demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", choices=("none", "platt", "isotonic"), default="platt"
    )
    parser.add_argument("--output", type=Path, default=Path("ensemble_roc_demo.png"))
    args = parser.parse_args()
    y_true, oof_preds_3d, oof_preds_tabular = synthetic_predictions(500, seed=42)
    y_test, test_preds_3d, test_preds_tabular = synthetic_predictions(500, seed=314)
    evaluate_ensembles(
        y_true,
        y_test,
        oof_preds_3d,
        oof_preds_tabular,
        test_preds_3d,
        test_preds_tabular,
        calibration=args.calibration,
        grid_step=0.01,
        roc_output_path=args.output,
        seed=42,
    )


if __name__ == "__main__":
    main()
