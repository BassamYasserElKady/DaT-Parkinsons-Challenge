#!/usr/bin/env python3
"""Evaluate an equal-weight soft-voting ensemble on the shared test split."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def load_predictions(path: Path) -> dict[str, tuple[int, float, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["uid"]: (int(row["actual"]), float(row["score_patient"]), int(row["predicted"])) for row in rows}


def metrics(name: str, truth: np.ndarray, scores: np.ndarray) -> dict[str, float | int | str]:
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "model": name,
        "n_test": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, predicted)),
        "auroc": float(roc_auc_score(truth, scores)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    resnet = load_predictions(root / "results/ants_registered_consecutive_resnet18_cbam/test_predictions.csv")
    densenet = load_predictions(root / "results/ants_registered_consecutive_densenet121/test_predictions.csv")
    uids = sorted(set(resnet) & set(densenet))
    if len(uids) != len(resnet) or len(uids) != len(densenet):
        raise ValueError("the two prediction files do not contain the same test UIDs")
    truth = np.asarray([resnet[uid][0] for uid in uids], dtype=int)
    if not np.array_equal(truth, np.asarray([densenet[uid][0] for uid in uids], dtype=int)):
        raise ValueError("the two prediction files disagree on labels")
    resnet_scores = np.asarray([resnet[uid][1] for uid in uids], dtype=float)
    densenet_scores = np.asarray([densenet[uid][1] for uid in uids], dtype=float)
    ensemble_scores = (resnet_scores + densenet_scores) / 2.0

    rows = [
        metrics("3D ResNet-18 + CBAM", truth, resnet_scores),
        metrics("DenseNet-121", truth, densenet_scores),
        metrics("Equal-weight soft-voting ensemble", truth, ensemble_scores),
    ]
    output = root / "results/ants_registered_resnet_densenet_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "actual", "resnet_score", "densenet_score", "ensemble_score", "ensemble_predicted"])
        writer.writeheader()
        for uid, label, resnet_score, densenet_score, ensemble_score in zip(uids, truth, resnet_scores, densenet_scores, ensemble_scores):
            writer.writerow({"uid": uid, "actual": int(label), "resnet_score": float(resnet_score), "densenet_score": float(densenet_score), "ensemble_score": float(ensemble_score), "ensemble_predicted": int(ensemble_score >= 0.5)})
    report_lines = [
        "# ResNet-18 and DenseNet-121 ensemble\n",
        "- Ensemble: equal-weight soft voting, averaging the two saved patient probabilities.",
        "- Evaluation: shared held-out test split of 409 scans; no test-set tuning.",
        "- The ResNet model is the native 3D ResNet-18 + CBAM model; DenseNet-121 is the 2D model adapted to use the 16 slices as channels.",
        "\n## Metrics\n",
    ]
    report_lines.extend(
        f"- **{row['model']}**: accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}, AUROC={row['auroc']:.4f}"
        for row in rows
    )
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['model']}: accuracy={row['accuracy']:.4f}, auroc={row['auroc']:.4f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
