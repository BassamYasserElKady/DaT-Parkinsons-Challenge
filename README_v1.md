# DaT Parkinson's Challenge

Reproducible code and documentation for binary classification of dopamine-transporter (DaT) SPECT scans as normal or dopaminergic-deficit.

This public code record excludes patient scans, labels, registered volumes, trained weights, predictions, caches, and submission archives.

## Pipeline

1. Read raw 3-D NIfTI DaT-SPECT volumes.
2. Rigidly register them to an MNI-space DAT template with ANTs.
3. Localize the striatum and select adaptive axial slices.
4. Prepare 3-D CNN inputs and an adaptive eight-slice axial MIP for Inception-v3.
5. Extract morphometric, intensity, shape, quarter-shape, and sliding local-intensity features.
6. Train ResNet-18 + CBAM, DenseNet-121, scratch Inception-v3, and an RBF-SVM.
7. Combine leakage-controlled out-of-fold logits with logistic-regression stacking.

## Main reported result

The strongest internally validated consistent pipeline was the four-model stack:

| Metric | Result |
|---|---:|
| AUROC | 0.9669 |
| Log loss | 0.2289 |
| Accuracy | 89.65% |

This is a five-fold out-of-fold result on 1,362 scans, not an external-dataset estimate.

## Reproducibility rule

Preprocessing, feature selection, SVM fitting, calibration, and ensemble-weight fitting must be performed inside each training fold. The validation fold remains untouched until inference.

## Contents

```text
scripts/       Registration, preprocessing, model, feature, CV, and packaging scripts
src/           Reusable project modules
configs/       Parameter configuration files
docs/          Design notes, audits, and experiment documentation
assets/        Small atlas metadata files only
requirements.txt
```

Most scripts expect local data and cached registrations. A clean reproduction requires obtaining the competition data separately and setting paths in the selected script.

Do not commit patient scans, labels, private submission data, model weights, prediction files, SSH keys, access tokens, or large generated caches.
