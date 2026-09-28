# Multi-sample design

## Scientific boundary

The project has 12 matched H&E + Xenium samples. Eight are development samples; four are final independent test samples. The initial trainable task is strictly:

```text
DeepSpot-M tile predictions -> calibration adapter -> calibrated Xenium-like tile expression
```

Bulk RNA-seq, microarray, and scRNA-seq measurements come from independent patients. They are not patient-specific measurements for these 12 samples and must not be supplied as conditioning vectors, covariates, or leakage-prone metadata to the first model. The model API therefore accepts only the tile prediction matrix.

## Manifest and geometry

`data/sample_manifest.csv` is the source of truth for sample IDs, split assignments, file paths, H&E dimensions, Xenium micron-per-pixel scale, tile size, and the row-major 3x3 affine transform from registration-scale H&E pixels to Xenium pixels. All data paths are relative to the project root.

Manifest validation is deliberately strict:

- required columns must exist;
- sample IDs must be unique;
- paths must be non-empty and remain below the project root;
- geometry values must be positive;
- the affine must have nine finite values and be invertible;
- there must be exactly 8 development and 4 final-test samples.

The validator checks metadata only. Parquet schemas are checked immediately before processing so a malformed input fails before alignment or model work begins.

## Per-sample preprocessing

For each row:

1. Read the listed DeepSpot-M and Xenium parquet files.
2. Require DeepSpot columns `spot`, `x`, `y`; treat all other non-metadata columns as candidate DeepSpot genes.
3. Require Xenium columns `feature_name`, `x_location`, `y_location`, and `qv`. Keep transcripts with `qv >= 20`. If `is_gene` exists, retain only true rows; otherwise remove the configured control prefixes `NegControl`, `Unassigned`, `BLANK`, and `NegPrb`. Decode byte-valued feature names.
4. Convert Xenium locations from microns to Xenium pixels by dividing by the sample's `xenium_um_per_pixel`.
5. Apply the inverse affine transform to homogeneous coordinates. The manifest transform is defined as registration-scale H&E pixels -> Xenium pixels.
6. Scale registration-scale H&E coordinates to full-resolution H&E using the sample's full/small width and height ratios. Reject points outside the full image bounds.
7. Assign `floor(coordinate / tile_size_px) * tile_size_px` for both axes. Match those tile coordinates to rounded DeepSpot `x,y` coordinates and retain only matched spots.
8. Count transcripts by `spot,feature_name`, pivot to a count matrix, and align rows to DeepSpot spots.
9. Compute the Xenium target as `log1p(CP10K)`:

   ```text
   log1p(10000 * counts / max(row_sum, 1e-8))
   ```

   Save raw DeepSpot predictions, log1p-CP10K targets, tile coordinates, split, and gene names in one compressed file per sample.

## Gene vocabulary and normalization

The gene vocabulary is the sorted intersection across the eight development samples only. Final-test samples are then required to contain every development-defined gene in that same order; missing genes cause an error rather than silently changing the test feature space. This prevents final-test expression values from influencing feature selection.

Normalization statistics are fit by concatenating only development-sample tiles. DeepSpot predictions and log1p-CP10K targets each receive gene-wise mean/standard-deviation normalization, with near-zero standard deviations replaced by 1. The statistics are persisted and applied unchanged to validation and final-test samples. No target-derived statistic from a final-test sample is used.

The saved processed matrices remain in the target's log1p-CP10K scale. Standardized arrays are a modeling concern and should be generated from the persisted development statistics by a future trainer.

## Sample-level splitting

There is no reuse of the legacy Cell 3 tile split as the final evaluation. Tiles from one sample must never be divided between model-selection groups in a way that makes the sample appear in both groups.

The required split contract is:

```text
development: sample_01 ... sample_10   # fit and grouped validation
```

Grouped validation creates folds from development sample IDs, not tile rows. A future trainer may use five grouped folds or another predeclared grouped scheme, but every validation fold must be sample-disjoint from its training fold. The leakage guard raises on any repeated sample ID across groups.

Model selection, hyperparameter tuning, early stopping, normalization fitting, gene-vocabulary derivation, threshold selection, and checkpoint selection must use development samples only. The final test set is evaluated once per sample and once pooled after the model is frozen.

## First model

`DeepSpotCalibrationAdapter` is a parameterized residual MLP:

```text
tile predictions -> Linear(G, hidden) -> GELU -> Dropout
                 -> Linear(hidden, latent) -> GELU
                 -> Linear(latent, hidden) -> GELU -> Dropout
                 -> Linear(hidden, G) correction
```

The output is a learned gene-wise affine baseline of the input plus the learned context correction. It has no bulk, microarray, scRNA-seq, patient ID, or patient embedding input. CUDA is selected automatically by `default_device()` when available; otherwise the model uses CPU.

The loss module supplies terms for a future trainer without running training:

- MSE over standardized prediction and target matrices;
- one minus mean gene-wise spatial Pearson across tiles;
- sampled pairwise within-tile gene-rank softplus loss;
- configurable weights for the Pearson and rank terms.

Any future trainer must document batch composition because Pearson is computed across the tile dimension. Loss weights, optimizer, epochs, and early stopping must be selected only within grouped development validation.

## Evaluation outputs

For each of the four final-test samples, report:

- number of tiles and genes;
- mean gene-wise spatial Pearson;
- mean per-tile gene-rank Spearman;
- aggregate abundance Pearson across genes;
- aggregate abundance Spearman across genes.

Then concatenate the four samples only for a separately labeled `POOLED_FINAL_TEST` row. Per-sample rows must not be replaced by a pooled number, since a pooled result can be dominated by tile count or a single sample. A future report should add uncertainty intervals across samples and preserve the individual gene-level tables.

