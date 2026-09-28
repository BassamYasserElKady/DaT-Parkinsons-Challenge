# Project summary

## Models

- ResNet-18 with CBAM on registered adaptive five-slice 3-D inputs.
- DenseNet-121 on the same registered 3-D representation.
- Scratch Inception-v3 on an adaptive eight-slice axial MIP.
- RBF-SVM using 82 guided morphometric and intensity-derived features.

## Evaluation

The main comparison used stratified five-fold out-of-fold predictions. Logistic regression was fit on training-fold logits and evaluated on the held-out fold. The pooled consistent guided stack reached AUROC 0.9669, log loss 0.2289, and accuracy 89.65%.

These metrics are development-cohort estimates and should not be described as external validation.

## Excluded artifacts

The public repository deliberately excludes all scans, labels, registered volumes, model checkpoints, per-scan predictions, cached arrays, and submission ZIP files.
