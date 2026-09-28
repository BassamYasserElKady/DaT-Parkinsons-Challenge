# Experiment history

This report records the development path using aggregate metrics only. Raw scans, labels, per-scan predictions, model weights, and generated caches are intentionally omitted.

## 1. Initial registered CNN ensemble

The project began with ANTs rigid registration into a DAT/MNI template, followed by adaptive registered-volume inputs for ResNet-18 and DenseNet-121. Initial held-out testing established a useful baseline, but the result was sensitive to the split and preprocessing details.

## 2. Adaptive slice and MIP experiments

We compared fixed and adaptive axial windows, different slice counts, and maximum-intensity projections. The most consistent CNN representation used adaptive five-slice 3-D inputs for the ResNet/DenseNet branches. The Inception branch used an adaptive eight-slice axial MIP resized to 299 x 299 pixels.

Gaussian smoothing, histogram-bin filtering, threshold variants, alternative crops, and non-MIP variants were evaluated as exploratory ablations. They were not retained unless their complete preprocessing and validation path matched the final pipeline.

## 3. Morphometric and intensity SVM

The RBF-SVM began with atlas-guided shape and intensity features. It was expanded to a consistent 82-feature vector containing:

- bilateral contour shape and morphometric measurements;
- 70/30 regional intensity summaries and asymmetry terms;
- bilateral full-shape and quarter-shape measurements;
- sliding 2 x 2 x 2 and 3 x 3 x 3 local-intensity summaries at multiple quantiles.

The final guided-feature SVM was evaluated with five-fold out-of-fold predictions. Hyperparameter tuning selected the RBF settings using inner-training folds and log loss.

## 4. Inception auxiliary-classifier ablation

Scratch Inception-v3 was tested with and without its auxiliary classifier using matched adaptive-eight-slice MIP inputs, 16 epochs, AdamW, cosine scheduling, and horizontal-flip augmentation. The auxiliary branch was retained for the earlier four-model stack because it had better matched OOF performance.

The no-auxiliary branch was also retained as a separately packaged comparison and must use its own stack calibration; auxiliary-model coefficients must not be reused for it.

## 5. Leakage-controlled ensemble stacking

The final comparison used stratified five-fold out-of-fold predictions. For every outer fold, the logistic meta-learner was fit only on the other folds' logits and then evaluated on the held-out fold. The final deployment stack was refit on all OOF rows after model selection.

### Aggregate results

| Component or stack | AUROC | Log loss | Accuracy |
|---|---:|---:|---:|
| Original four-model stack | 0.9638 | 0.2421 | 89.13% |
| Consistent guided 82-feature four-model stack | 0.9669 | 0.2289 | 89.65% |
| Tuned guided-SVM four-model stack | 0.9665 | 0.2295 | 90.09% |
| Consistent guided SVM alone | 0.9534 | 0.2665 | 87.59% |
| Tuned guided SVM alone | 0.9547 | 0.2617 | 87.74% |
| Auxiliary Inception-v3 alone | 0.9628 | 0.2559 | 88.91% |
| No-auxiliary Inception-v3 alone | 0.9599 | 0.2657 | 88.40% |

The strongest consistent internal result was the guided four-model stack with AUROC 0.9669 and log loss 0.2289. These are development-cohort OOF estimates, not external-dataset results.

## 6. Packaging and runtime validation

Submission packages were smoke-tested from raw NIfTI input through rigid registration, preprocessing, all model branches, SVM feature extraction, logistic stacking, and submission CSV generation. The smoke tests verified runtime integrity only; they do not replace external validation.

## Interpretation

The main improvement came from making the registration, feature extraction, fold alignment, and stack calibration internally consistent. Several earlier attractive numbers came from exploratory or mismatched pipelines and are not treated as final evidence.
