#!/usr/bin/env python3
"""Cross-fitted log-loss blending of the fresh ResNet, DenseNet, and RBF-SVM OOF predictions."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CNN = ROOT / "results/repro_seed20260932/cnn/oof_predictions.csv"
SVM = ROOT / "results/repro_seed20260932/svm/oof_predictions.csv"
OUT = ROOT / "results/repro_seed20260932/three_model_blend"
EPS = 1e-6


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip(p)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def beta_x(p: np.ndarray) -> np.ndarray:
    p = clip(p)
    return np.column_stack((np.log(p), -np.log1p(-p)))


def fit_calibrator(kind: str, p: np.ndarray, y: np.ndarray):
    p = clip(p)
    if kind == "none":
        return lambda q: clip(q)
    if kind == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        model.fit(logit(p)[:, None], y)
        return lambda q: clip(model.predict_proba(logit(q)[:, None])[:, 1])
    if kind == "beta":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        model.fit(beta_x(p), y)
        return lambda q: clip(model.predict_proba(beta_x(q))[:, 1])
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return lambda q: clip(model.predict(clip(q)))
    raise ValueError(kind)


def combine(p: np.ndarray, weights: np.ndarray, space: str) -> np.ndarray:
    if space == "probability":
        return clip(p @ weights)
    return clip(sigmoid(np.column_stack([logit(p[:, i]) for i in range(p.shape[1])]) @ weights))


def optimize_weights(p: np.ndarray, y: np.ndarray, space: str, style: str) -> np.ndarray:
    if style == "equal_cnn_plus_svm":
        cnn = 0.5 * (p[:, 0] + p[:, 1])
        pair = np.column_stack((cnn, p[:, 2]))
        result = minimize_scalar(
            lambda w: log_loss(y, combine(pair, np.array([w, 1.0 - w]), space)),
            bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10},
        )
        return np.array([result.x / 2.0, result.x / 2.0, 1.0 - result.x])
    result = minimize(
        lambda w: log_loss(y, combine(p, w, space)),
        x0=np.full(3, 1.0 / 3.0), method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, clip(p))),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "incorrect": int(np.sum((p >= 0.5) != y)),
    }


def read_inputs() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with CNN.open(newline="", encoding="utf-8") as handle:
        cnn_rows = list(csv.DictReader(handle))
    with SVM.open(newline="", encoding="utf-8") as handle:
        svm_by_uid = {row["uid"]: row for row in csv.DictReader(handle)}
    uids, y, folds, values = [], [], [], []
    for row in cnn_rows:
        other = svm_by_uid[row["uid"]]
        if int(row["actual"]) != int(other["actual"]) or int(row["fold"]) != int(other["fold"]):
            raise RuntimeError(f"Prediction mismatch for {row['uid']}")
        uids.append(row["uid"])
        y.append(int(row["actual"]))
        folds.append(int(row["fold"]))
        values.append([float(row["resnet"]), float(row["densenet"]), float(other["svm"])])
    return uids, np.asarray(y), np.asarray(folds), np.asarray(values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uids, y, folds, raw = read_inputs()
    kinds = ("none", "platt", "beta", "isotonic")
    configurations = list(itertools.product(
        kinds, ("probability", "logit"),
        ("equal_cnn_plus_svm", "free_three_weights"), kinds,
    ))
    rankings, predictions, parameters = [], {}, {}
    for pre, space, style, post in configurations:
        name = f"pre-{pre}__{space}__{style}__post-{post}"
        oof = np.full(len(y), np.nan)
        fold_params = []
        for fold in range(1, 6):
            train, test = folds != fold, folds == fold
            train_p = np.empty((train.sum(), 3)); test_p = np.empty((test.sum(), 3))
            for j in range(3):
                calibrate = fit_calibrator(pre, raw[train, j], y[train])
                train_p[:, j] = calibrate(raw[train, j])
                test_p[:, j] = calibrate(raw[test, j])
            weights = optimize_weights(train_p, y[train], space, style)
            train_blend = combine(train_p, weights, space)
            test_blend = combine(test_p, weights, space)
            post_calibrate = fit_calibrator(post, train_blend, y[train])
            oof[test] = post_calibrate(test_blend)
            fold_params.append({"fold": fold, "resnet": float(weights[0]),
                                "densenet": float(weights[1]), "svm": float(weights[2])})
        result = metric(y, oof)
        mean_w = np.mean([[r["resnet"], r["densenet"], r["svm"]] for r in fold_params], axis=0)
        rankings.append({"pipeline": name, "pre": pre, "space": space, "style": style,
                         "post": post, "resnet_weight": float(mean_w[0]),
                         "densenet_weight": float(mean_w[1]), "svm_weight": float(mean_w[2]), **result})
        predictions[name] = oof
        parameters[name] = fold_params
    rankings.sort(key=lambda row: row["log_loss"])
    winner = rankings[0]
    best = predictions[winner["pipeline"]]
    with (OUT / "pipeline_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rankings[0])); writer.writeheader(); writer.writerows(rankings)
    with (OUT / "best_oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["uid", "actual", "fold", "prediction"])
        writer.writerows(zip(uids, y, folds, best))
    with (OUT / "winner_fold_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "resnet", "densenet", "svm"])
        writer.writeheader(); writer.writerows(parameters[winner["pipeline"]])
    summary = {"seed": 20260932, "n": len(y), "standalone": {
        "resnet": metric(y, raw[:, 0]), "densenet": metric(y, raw[:, 1]), "svm": metric(y, raw[:, 2]),
        "equal_cnn": metric(y, 0.5 * (raw[:, 0] + raw[:, 1]))},
        "configurations_tested": len(rankings), "winner": winner,
        "winner_fold_weights": parameters[winner["pipeline"]], "top_5": rankings[:5]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
