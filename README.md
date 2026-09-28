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

## How the final model was reached

The final result was built through a sequence of controlled experiments rather than a single training run:

| Stage | What was tested | Outcome |
|---|---|---|
| Registration baseline | ANTs rigid registration into a DAT/MNI template | Established a common physical space |
| CNN baseline | ResNet-18 and DenseNet-121 on registered 3-D volumes | Provided strong image-only predictors |
| Slice selection | Fixed versus adaptive axial windows | Adaptive windows handled variable scan coverage better |
| Projection comparison | MIP versus non-MIP representations | Adaptive 8-slice axial MIP was retained for Inception-v3 |
| Inception ablation | Auxiliary versus no auxiliary classifier | Auxiliary Inception performed better in the matched OOF comparison |
| Morphometric SVM | Shape, intensity, SBR, asymmetry, and regional features | Added complementary information to the CNNs |
| Guided SVM expansion | 82 consistent features with full shapes, quarters, and sliding local-intensity summaries | Improved the internally consistent ensemble |
| SVM tuning | RBF `C`, `gamma`, and class-weight search inside nested folds | Slightly improved standalone SVM performance |
| Ensemble comparison | Equal averaging, manual weights, calibration, routing, and logit stacking | Leakage-controlled logistic logit stacking was retained |
| Runtime validation | Raw NIfTI through registration, four branches, SVM extraction, and submission CSV | Confirmed the inference path end to end |

### Aggregate development results

| Model or ensemble | AUROC | Log loss | Accuracy |
|---|---:|---:|---:|
| Original four-model stack | 0.9638 | 0.2421 | 89.13% |
| Consistent guided 82-feature stack | **0.9669** | **0.2289** | 89.65% |
| Tuned guided-SVM stack | 0.9665 | 0.2295 | **90.09%** |
| Guided SVM alone | 0.9534 | 0.2665 | 87.59% |
| Tuned guided SVM alone | 0.9547 | 0.2617 | 87.74% |
| Auxiliary Inception-v3 alone | 0.9628 | 0.2559 | 88.91% |
| No-auxiliary Inception-v3 alone | 0.9599 | 0.2657 | 88.40% |

## Final private leaderboard results

| Evaluation set | AUROC | Log loss | Rank |
|---|---:|---:|---:|
| First private set | 0.9622 | 0.2414 | 20th |
| Second private set | 0.9421 | 0.2972 | 35th |

## Reference leaderboard comparison

The following list is reproduced from the referenced third-place solution's project documentation. `Djehuty` is appended using the two final scores recorded above.

| team | public | private | move |
|---|---|---|---:|
| Marc-Dvci | 1st, 0.2154 | 28th, 0.2931 | −27 |
| TheAvengers | 2nd, 0.2249 | 1st, 0.2532 | +1 |
| South-Wing | 3rd, 0.2263 | 17th, 0.2848 | −14 |
| this repo | 7th, 0.2313 | 6th, 0.2732 | +1 |
| venkt | 10th, 0.2340 | 38th, 0.2971 | −28 |
| Tigertech | 12th, 0.2363 | 2nd, 0.2647 | +10 |
| **Djehuty** | **20th, 0.2414** | **35th, 0.2972** | **—** |

The detailed sanitized history is in [`docs/experiment_history.md`](docs/experiment_history.md). Aggregate development values are in [`reports/aggregate_metrics.json`](reports/aggregate_metrics.json), and the private leaderboard values are in [`reports/competition_scores.json`](reports/competition_scores.json). Internal OOF scores are not external leaderboard scores.

## Reproducibility rule

Preprocessing, feature selection, SVM fitting, calibration, and ensemble-weight fitting must be performed inside each training fold. The validation fold remains untouched until inference.

## Contents

```text
scripts/       Registration, preprocessing, model, feature, CV, and packaging scripts
src/           Reusable project modules
configs/       Parameter configuration files
docs/          Design notes, audits, and experiment documentation
reports/       Sanitized aggregate metrics only
assets/        Small atlas metadata files only
requirements.txt
```

Most scripts expect local data and cached registrations. A clean reproduction requires obtaining the competition data separately and setting paths in the selected script.

Do not commit patient scans, labels, private submission data, model weights, prediction files, SSH keys, access tokens, or large generated caches.
