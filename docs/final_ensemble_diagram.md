# Final ensemble architecture

```mermaid
flowchart LR
    A[Raw 3-D DaT-SPECT NIfTI] --> B[ANTs rigid registration]
    B --> C[Registered volume]

    C --> D[Adaptive 5-slice 3-D input]
    C --> E[Adaptive 8-slice axial MIP]
    C --> F[82 guided morphometric and intensity features]

    D --> G[ResNet-18 + CBAM]
    D --> H[DenseNet-121]
    E --> I[Scratch Inception-v3]
    F --> J[Standardized RBF-SVM]

    G --> K[Model probabilities]
    H --> K
    I --> K
    J --> K
    K --> L[Logit transformation]
    L --> M[Leakage-controlled logistic stack]
    M --> N[Final dopaminergic-deficit probability]
```

The CNN and SVM branches are trained independently. Their out-of-fold logits are combined by a logistic-regression meta-learner. During deployment, the same preprocessing and four model branches produce one final probability per scan.
