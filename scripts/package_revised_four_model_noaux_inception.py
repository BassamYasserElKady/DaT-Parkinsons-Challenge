"""Build the revised four-model submission package using no-auxiliary Inception-v3.

The logistic stack is cross-fitted from fold-matched OOF predictions so its
reported validation metrics remain leakage-controlled.  Final deployment
weights are fit on all OOF rows, which is appropriate after model selection.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_ensemble_revised_aux_guided_four_model_tuned_svm"
PACKAGE = ROOT / "submission_ensemble_revised_noaux_guided_four_model_tuned_svm"
ZIP_PATH = ROOT / "submission_ensemble_revised_noaux_guided_four_model_tuned_svm.zip"
TUNED = ROOT / "results" / "stack_auxiliary_inception_tuned_svm_oof.npz"
NOAUX = ROOT / "results" / "cv5_inception8_mip_noauxiliary_matched" / "oof_predictions.npz"
FULL_INCEPTION = ROOT / "results" / "full_inception8_mip_submission" / "inception8_mip_weights.pt"
OUT = ROOT / "results" / "stack_noaux_inception_tuned_svm"
EPS = 1e-6


def logits(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "incorrect": int(np.count_nonzero((p >= 0.5) != y)),
    }


def main() -> None:
    tuned = np.load(TUNED)
    noaux = np.load(NOAUX)
    y = np.asarray(tuned["y"], dtype=int)
    fold = np.asarray(tuned["fold"], dtype=int)
    if not np.array_equal(y, np.asarray(noaux["y"], dtype=int)):
        raise RuntimeError("No-auxiliary Inception labels do not align with stack OOF labels")

    # Preserve the tuned-SVM package's ResNet, DenseNet, and SVM OOF branches;
    # replace only the Inception branch with its matched no-auxiliary OOF output.
    components = np.column_stack(
        [tuned["resnet"], tuned["densenet"], tuned["tuned_svm"], noaux["prediction"]]
    )
    x = logits(components)
    oof = np.empty(len(y), dtype=float)
    reports: list[dict[str, float | int]] = []
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        meta = LogisticRegression(C=1.0, max_iter=2000, random_state=20260931)
        meta.fit(x[train], y[train])
        oof[test] = meta.predict_proba(x[test])[:, 1]
        reports.append({"fold": int(f), **metrics(y[test], oof[test])})

    final_meta = LogisticRegression(C=1.0, max_iter=2000, random_state=20260931)
    final_meta.fit(x, y)
    metadata = {
        "architecture": "resnet18_cbam_3d_densenet121_3d_consistent_guided_82_feature_rbf_svm_scratch_inception_v3_adaptive8_mip_logit_stack",
        "training_scan_count": 1362,
        "cnn_preprocessing": {"adaptive_slices": 5, "gaussian_sigma": 0.0, "input_shape": [16, 64, 64]},
        "inception_preprocessing": {"adaptive_slices": 8, "mip": True, "input_shape": [1, 299, 299], "gaussian_sigma": 0.0},
        "training": {
            "resnet_epochs": 16,
            "densenet_epochs": 14,
            "inception_epochs": 16,
            "inception_auxiliary_classifier": False,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "augmentation": "random_horizontal_flip_only",
        },
        "logit_stack": {
            "models": ["resnet", "densenet", "guided_svm", "inception8_mip_noauxiliary"],
            "coef": final_meta.coef_[0].tolist(),
            "intercept": float(final_meta.intercept_[0]),
            "oof_source": "five-fold leakage-controlled predictions; no-auxiliary Inception branch",
        },
        "guided_svm": {"feature_count": 82, "tuned_full_model": {"C": 3.0, "gamma": "scale", "class_weight": "balanced"}, "registration_cache": "ants_registration_mni"},
    }
    summary = {
        "evaluation": "five-fold leakage-controlled cross-fitted logistic stacking",
        "components": {"resnet": metrics(y, components[:, 0]), "densenet": metrics(y, components[:, 1]), "tuned_guided_svm": metrics(y, components[:, 2]), "inception8_mip_noauxiliary": metrics(y, components[:, 3])},
        "four_model_stack": metrics(y, oof),
        "folds": reports,
        "deployment_stack_coef": metadata["logit_stack"]["coef"],
        "deployment_stack_intercept": metadata["logit_stack"]["intercept"],
    }

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    shutil.copytree(SOURCE, PACKAGE)
    shutil.copy2(FULL_INCEPTION, PACKAGE / "assets" / "model" / "inception8_mip_noauxiliary_weights.pt")
    old_weight = PACKAGE / "assets" / "model" / "auxiliary_inception8_mip_weights.pt"
    if old_weight.exists():
        old_weight.unlink()
    model_py = PACKAGE / "src" / "model.py"
    main_py = PACKAGE / "main.py"
    model_py.write_text(model_py.read_text().replace("AuxiliaryInceptionMIP", "InceptionMIP").replace("strict=False", "strict=True"))
    main_py.write_text(main_py.read_text().replace("auxiliary_inception8_mip_weights.pt", "inception8_mip_noauxiliary_weights.pt"))
    (PACKAGE / "assets" / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(OUT / "oof_predictions.npz", y=y, fold=fold, components=components, stack=oof)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", ROOT, PACKAGE.name)
    print(json.dumps({"summary": summary, "package": str(PACKAGE), "zip": str(ZIP_PATH), "zip_bytes": ZIP_PATH.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
