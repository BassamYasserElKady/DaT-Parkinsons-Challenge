#!/usr/bin/env python3
"""Five-fold ablation of explicit occipital/SBR features for the RBF SVM."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "results/cv5_svm_occipital_sbr/features.npz"
LABELS = ROOT / "results/ants_registered_consecutive_resnet18_cbam/split.csv"
OUT = ROOT / "results/cv5_svm_occipital_sbr/ablation.json"
SEED = 20260931

def metric(y, p):
    q = p >= .5
    return {"auroc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy": float(accuracy_score(y, q)), "incorrect": int(np.sum(q != y))}

def main():
    z = np.load(FEATURES)
    x = z["occipital_sbr32"]
    labels = __import__("pandas").read_csv(LABELS, dtype={"uid": str}).sort_values("uid")
    y = labels["is_pathologic"].to_numpy(dtype=int)
    if x.shape != (len(y), 32):
        raise RuntimeError(f"Unexpected feature shape {x.shape}")
    groups = {
        "baseline_22": [],
        "occipital_mean": [22],
        "occipital_std": [23],
        "four_regional_sbrs": [24, 25, 26, 27],
        "mean_caudate_putamen_sbr": [28, 29],
        "sbr_asymmetries": [30, 31],
        "occipital_plus_sbr": list(range(22, 32)),
    }
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {}
    for name, extra in groups.items():
        cols = list(range(22)) + extra
        p = np.full(len(y), np.nan)
        folds = []
        for fold, (train, test) in enumerate(splitter.split(x, y), 1):
            model = make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale", probability=True,
                                                        random_state=SEED + fold))
            model.fit(x[train][:, cols], y[train])
            p[test] = model.predict_proba(x[test][:, cols])[:, 1]
            folds.append({"fold": fold, **metric(y[test], p[test])})
        results[name] = {"feature_count": len(cols), "pooled": metric(y, p), "folds": folds}
        print(name, json.dumps(results[name]["pooled"]), flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
