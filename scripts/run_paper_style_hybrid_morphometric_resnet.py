#!/usr/bin/env python3
"""Combine morphometric and frozen 3D ResNet features with paper-style selection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from train_registered_consecutive_resnet18_cbam import ResNet18CBAM3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rf-repeats", type=int, default=100)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--max-top-features", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_embeddings(root: Path, uids: list[str], device: torch.device) -> np.ndarray:
    model = ResNet18CBAM3D().to(device)
    checkpoint = torch.load(root / "results/ants_registered_consecutive_resnet18_cbam/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inputs = np.load(root / "results/ants_registered_consecutive_resnet18_cbam/input_windows.npy", mmap_mode="r")
    selection_path = root / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
    with selection_path.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    split_uids = [row["uid"] for row in split_rows]
    positions = {uid: i for i, uid in enumerate(split_uids)}
    embeddings = np.empty((len(uids), 512), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(uids), 32):
            batch_uids = uids[start:start + 32]
            batch = torch.from_numpy(np.asarray(inputs[[positions[uid] for uid in batch_uids]])).unsqueeze(1).to(device)
            x = model.stem(batch)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            embeddings[start:start + len(batch_uids)] = model.pool(x).flatten(1).cpu().numpy()
    return embeddings


def select_features(X: np.ndarray, y: np.ndarray, names: list[str], args: argparse.Namespace, output: Path) -> tuple[np.ndarray, list[str], dict[str, object]]:
    variance = np.var(X, axis=0)
    keep_variance = variance >= 0.01
    X1 = X[:, keep_variance]
    names1 = [name for name, keep in zip(names, keep_variance) if keep]

    f_scores, p_values = f_classif(X1, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    keep_anova = p_values < 0.01
    if keep_anova.sum() < 2:
        keep_anova[np.argsort(-f_scores)[: min(10, len(keep_anova))]] = True
    X2 = X1[:, keep_anova]
    names2 = [name for name, keep in zip(names1, keep_anova) if keep]
    f2 = f_scores[keep_anova]

    corr = np.corrcoef(X2, rowvar=False) if X2.shape[1] > 1 else np.ones((1, 1))
    order = np.argsort(-f2)
    kept_indices: list[int] = []
    for index in order:
        if not kept_indices or np.max(np.abs(corr[index, kept_indices])) <= 0.85:
            kept_indices.append(int(index))
    X3 = X2[:, kept_indices]
    names3 = [names2[index] for index in kept_indices]

    importances = np.zeros(X3.shape[1], dtype=float)
    for repeat in range(args.rf_repeats):
        forest = RandomForestClassifier(
            n_estimators=args.rf_trees,
            random_state=args.random_state + repeat,
            class_weight="balanced",
            n_jobs=1,
        )
        forest.fit(X3, y)
        importances += forest.feature_importances_
        if (repeat + 1) % 10 == 0 or repeat + 1 == args.rf_repeats:
            print(f"random_forest_repeats={repeat + 1}/{args.rf_repeats}", flush=True)
    importances /= args.rf_repeats
    ranking = np.argsort(-importances)
    ranked_names = [names3[index] for index in ranking]

    cv = StratifiedKFold(5, shuffle=True, random_state=args.random_state)
    max_k = min(args.max_top_features, len(ranking))
    curve = []
    for k in range(1, max_k + 1):
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        scores = cross_val_score(model, X3[:, ranking[:k]], y, cv=cv, scoring="accuracy", n_jobs=-1)
        curve.append({"k": k, "accuracy_mean": float(scores.mean()), "accuracy_std": float(scores.std())})
        if k % 10 == 0 or k == max_k:
            print(f"feature_curve_k={k}/{max_k}", flush=True)
    best_accuracy = max(row["accuracy_mean"] for row in curve)
    best_k = min(int(row["k"]) for row in curve if row["accuracy_mean"] == best_accuracy)
    selected_indices = ranking[:best_k]
    selected_names = [names3[index] for index in selected_indices]

    write_rows(output / "feature_selection_curve.csv", curve, ["k", "accuracy_mean", "accuracy_std"])
    write_rows(
        output / "feature_importance_ranking.csv",
        [{"rank": rank + 1, "feature": names3[index], "mean_rf_importance": float(importances[index])} for rank, index in enumerate(ranking)],
        ["rank", "feature", "mean_rf_importance"],
    )
    details = {
        "initial_features": len(names),
        "after_variance": int(keep_variance.sum()),
        "after_anova": int(keep_anova.sum()),
        "after_correlation": len(names3),
        "selected_count": best_k,
        "selected_cv_accuracy": best_accuracy,
        "selected_names": ";".join(selected_names),
    }
    return selected_indices, selected_names, details


def evaluate_model(X_fit: np.ndarray, y_fit: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    model.fit(X_fit, y_fit)
    score = model.predict_proba(X_test)[:, 1]
    prediction = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_test, prediction)),
        "auroc": float(roc_auc_score(y_test, score)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    return metrics, prediction, score


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/hybrid_morphometric_resnet18_cbam_paper"
    output.mkdir(parents=True, exist_ok=True)
    morph_path = root / "results/ants_registered_top025_morphometric_medium_gaussian_svm/morphometric_features.csv"
    morph = pd.read_csv(morph_path)
    morph_names = [
        "left_area", "left_perimeter", "left_circularity", "left_major_axis_length", "left_minor_axis_length", "left_eccentricity",
        "right_area", "right_perimeter", "right_circularity", "right_major_axis_length", "right_minor_axis_length", "right_eccentricity",
    ]
    morph = morph.sort_values("uid").reset_index(drop=True)
    uids = morph["uid"].astype(str).tolist()
    y = morph["is_pathologic"].astype(int).to_numpy()
    morph_features = morph[morph_names].to_numpy(dtype=np.float32)
    split = pd.read_csv(root / "results/ants_registered_consecutive_resnet18_cbam/split.csv").set_index("uid").loc[uids]["split"].to_numpy()
    fit_idx = np.flatnonzero(split != "test")
    test_idx = np.flatnonzero(split == "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep = extract_embeddings(root, uids, device)

    deep_names = [f"deep_feat_{i + 1}" for i in range(deep.shape[1])]
    feature_names = [f"morph_{name}" for name in morph_names] + deep_names
    combined = np.concatenate([morph_features, deep], axis=1)
    np.save(output / "combined_features.npy", combined)
    with (output / "combined_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "is_pathologic"] + feature_names)
        writer.writerows([[uid, int(label), *row] for uid, label, row in zip(uids, y, combined)])

    selected_local, selected_names, selection_details = select_features(combined[fit_idx], y[fit_idx], feature_names, args, output)
    # selected_local indexes the post-correlation matrix, so recover selected features from the ranking file.
    ranking = pd.read_csv(output / "feature_importance_ranking.csv")
    selected_names = ranking.head(int(selection_details["selected_count"]))["feature"].tolist()
    selected_global = np.asarray([feature_names.index(name) for name in selected_names], dtype=int)
    X = combined[:, selected_global]
    model_metrics, predictions, scores = evaluate_model(X[fit_idx], y[fit_idx], X[test_idx], y[test_idx])
    metrics = {
        "model": "paper_style_hybrid_linear_discriminant",
        "device_for_embedding": str(device),
        "n_scans": len(y),
        "n_fit": len(fit_idx), "n_test": len(test_idx),
        "patients_total": int(y.sum()), "healthy_total": int((y == 0).sum()),
        "patients_test": int(y[test_idx].sum()), "healthy_test": int((y[test_idx] == 0).sum()),
        **selection_details,
        **model_metrics,
    }
    write_rows(output / "test_metrics.csv", [metrics], list(metrics))
    write_rows(
        output / "test_predictions.csv",
        [{"uid": uids[index], "actual": int(y[index]), "predicted": int(predictions[position]), "score_patient": float(scores[position])} for position, index in enumerate(test_idx)],
        ["uid", "actual", "predicted", "score_patient"],
    )
    (output / "selected_features.txt").write_text("\n".join(selected_names) + "\n", encoding="utf-8")
    report = output / "report.md"
    report.write_text(
        "# Paper-style morphometric + 3D ResNet hybrid\n\n"
        "The hybrid combines the 12 registered top-0.25% morphometric features with 512 frozen penultimate-layer features from the trained 3D ResNet-18 + CBAM model. Feature selection is fitted only on the non-test portion: variance filtering, ANOVA (p<0.01), correlation pruning (|r|>0.85), 100 repeated Random Forest importance rankings, and a 5-fold LDA feature-count curve. The final classifier is standardized Linear Discriminant Analysis, matching the paper’s reported hybrid classifier.\n\n"
        + "\n".join(f"- **{key}:** {value:.4f}" if isinstance(value, float) else f"- **{key}:** {value}" for key, value in metrics.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"metrics={metrics}")
    print(f"selected_features={selected_names}")


if __name__ == "__main__":
    main()
