# Legacy baseline audit

Source audited, read-only: `notebooks/legacy/extracted_deepspot_calibration_5_cells.py`

This is a notebook-exported, single-specimen proof of concept. Its observed run produced 4,896 matched tiles and a 305-gene intersection, with `SAMPLE_ID = "sample_1"`.

## 1. Expected inputs and required columns

The code expects these three files at the following exact Kaggle paths:

| Input | Format | Required columns | Role |
|---|---|---|---|
| `/kaggle/input/datasets/bassamyasser/xenium-spatial-predictions/deepspot_prediction_wide.parquet` | Parquet | `spot`, `x`, `y` | DeepSpot-M wide tile predictions. Every other column except `patient`, `spot`, `x`, and `y` is treated as a candidate gene column. |
| `/kaggle/input/datasets/bassamyasser/xenium-spatial-predictions/transcripts.parquet` | Parquet | `feature_name`, `x_location`, `y_location`, `qv` | Xenium transcript-level observations. `is_gene` is optional; if absent, names beginning `NegControl`, `Unassigned`, `BLANK`, or `NegPrb` are removed instead. |
| `/kaggle/input/datasets/bassamyasser/bulks-csv/bulk.csv` | CSV | `patient`, `gene`, `bulk_value` | Bulk expression used as the gene-wise input vector. |

Additional assumptions are implicit: DeepSpot coordinates `x`/`y` are tile-grid coordinates; transcript locations are Xenium microns; `feature_name` may need UTF-8 decoding from bytes; gene names must match exactly across tables after decoding; and the bulk table must contain at least one usable row for the selected sample and shared genes. The code averages duplicate bulk rows by gene and fills missing reindexed values with the median.

No H&E image file is read. H&E-to-Xenium registration is represented only by a hard-coded transform and image dimensions.

## 2. H&E/Xenium coordinate alignment

The procedure is:

1. Filter transcripts to `qv >= 20`. Keep `is_gene == True` when that column exists; otherwise remove the named control/unassigned prefixes.
2. Convert Xenium coordinates from microns to Xenium morphology pixels using `0.2125` microns per pixel:
   `xp = x_location / 0.2125`, `yp = y_location / 0.2125`.
3. Form homogeneous points `[xp, yp, 1]`.
4. Apply the inverse of the matrix declared as small-H&E-pixels-to-Xenium-pixels:

   ```text
   [[ 3.424931528,  0.017012938, -6835.558645],
    [ 0.017012938, -3.424931528, 32660.661210],
    [ 0.0,         0.0,            1.0     ]]
   ```

   The resulting homogeneous coordinates are divided by their third component to obtain `small_x` and `small_y`.
5. Scale small-H&E pixels to the full-resolution DeepSpot H&E coordinate system:
   `he_x_full = small_x * (27587 / 13793)` and
   `he_y_full = small_y * (20511 / 10255)`.
6. Retain points inside `[0, 27587) x [0, 20511)`.
7. Assign the global 224-pixel tile containing each point:
   `tile_x = floor(he_x_full / 224) * 224` and
   `tile_y = floor(he_y_full / 224) * 224`.
8. Build a lookup from rounded DeepSpot `(x, y)` to `spot` using distinct rows of `ds[['spot', 'x', 'y']]`. Keep only transcripts whose tile coordinate has a DeepSpot spot.
9. Count retained transcripts by `(spot, feature_name)` and pivot to a spot-by-gene matrix.

The source notes that coordinate `13664` is divisible by 224, so the two presumed GPU halves use one global tile grid. That is an assumption to validate during migration, not a general registration guarantee. The transform, dimensions, pixel size, axis orientation, rounding behavior, and bounds should be treated as calibration metadata and independently validated with landmarks or overlays.

## 3. Normalization and the 305-gene intersection

### Intersection and matrix construction

The DeepSpot candidate genes are all columns other than `patient`, `spot`, `x`, and `y`. Xenium genes are the pivoted `feature_name` columns. The code takes the sorted set intersection:

```text
genes = sorted(set(deep_genes) & set(truth.columns))
```

It then indexes both matrices by the DeepSpot spots that occur in the Xenium truth matrix, fills absent spot/gene counts with zero, and materializes:

```text
P        = DeepSpot values, float32
Y_counts = Xenium transcript counts, float32
```

The original run reports `P.shape == (4896, 305)` and `len(genes) == 305`. The 305 is therefore an observed intersection for this input, not a universal fixed gene list. The source only asserts that the intersection is greater than 20.

### Xenium and bulk normalization

For each spot, Xenium counts are library-size normalized to 10,000 and transformed:

```text
Y = log1p(10000 * Y_counts / clip(row_sum, 1e-8, None))
```

Bulk rows are filtered to `patient == "sample_1"` and the shared genes. If that returns no rows and the bulk file has exactly one unique patient, the sole patient is used as a fallback. Duplicate gene rows are averaged, genes are reindexed to the shared-gene order, missing values are filled with the median, values are clipped at zero, then normalized and transformed:

```text
b = log1p(10000 * clip(bulk_gene_value, 0, None) / max(sum, 1e-8))
```

`B` is a copy of this same vector for every tile; it is constant across the entire one-sample dataset.

### Z-scoring

The DeepSpot matrix and Xenium matrix are independently standardized gene-by-gene using means and standard deviations computed on the relevant training subset. Standard deviations below `1e-6` are replaced by 1. The adapter receives a bulk vector standardized across genes using its global mean and standard deviation plus `1e-8`.

The code has three related statistics regimes:

- Cell 1: statistics from the initial spatial-block training set.
- Optuna: statistics from the inner training set for tuning, then recomputed from the complete 80% development set for final retraining.
- Spatial holdout: statistics from the spatial-training portion only.

This train-only fitting principle is reusable, but it must be enforced in a clean pipeline and persisted with the model.

## 4. Adapter architecture and loss terms

The adapter is called `OptunaBulkToXenium` in the tuning/final-random-test section and `SpatialBestConverter` in the spatial-holdout section. The two implementations have the same architecture.

Let `G = 305`, `h` be the hidden width, `z` the latent width, `P_i` be a standardized DeepSpot tile vector, and `b` be the standardized bulk vector. The architecture is:

```text
tile_encoder: Linear(G, h) -> ReLU -> Dropout(p) -> Linear(h, z) -> ReLU
scale_head:   Linear(z, h) -> ReLU -> Dropout(p) -> Linear(h, G)
scale_i       = 2 * tanh(scale_head(tile_encoder(P_i)))
```

There are two learned gene-wise parameter vectors:

- `gene_intercept`, initialized to zeros;
- `gene_bulk_gain`, initialized to ones.

The output is:

```text
Xenium_hat[i, g] = gene_intercept[g]
                    + scale[i, g] * gene_bulk_gain[g] * b[g]
```

Thus bulk gene `g` enters only output gene `g`; the DeepSpot tile context predicts a bounded, tile-specific scale for each gene. The model does not directly add a DeepSpot residual to the prediction.

The best Optuna configuration recorded in the source is: latent 96, hidden 128, dropout `0.32539417831990053`, learning rate `0.0015620386984526499`, weight decay `0.00022026297256779741`, Pearson weight `0.4979229468075753`, rank weight `0.14760016020610808`, scale weight `4.1292707579765786e-05`, and 210 final epochs.

### Training loss

The tuning and final adapter training objective is:

```text
L = L_MSE
    + alpha * L_spatial_Pearson
    + beta * L_pairwise_rank
    + lambda_scale * L_scale
```

All four terms are present in both adapter training loops:

1. `L_MSE`: mean squared error over predicted and standardized Xenium matrices.
2. `L_spatial_Pearson`: `1 - mean_g(r_g)`, where Pearson correlation is computed across tiles independently for each gene. The implementation replaces NaN/inf correlations with zero before averaging.
3. `L_pairwise_rank`: sample 3,000 random unequal gene-index pairs, compute prediction and target differences within every tile, discard target ties, and average `softplus(-sign(target_difference) * prediction_difference)`. This encourages within-tile gene ordering.
4. `L_scale`: mean of `scale ** 2`, penalizing large context-dependent gains.

The optimizer is AdamW and gradients are clipped to norm 5. The validation selection score is separate from the training loss: mean gene-wise spatial Pearson plus `0.25 *` mean per-tile gene Spearman. Optuna maximizes that score, with 40 TPE trials, median pruning, and early stopping patience of 30 validation-score units measured in five-epoch checks.

The preliminary baseline metric, used before adapter training, is a per-gene test loss `MSE + 0.10 * (1 - Pearson)`, averaged over genes after randomly selecting 80% of the initial 832 held-out tiles. It is an evaluation calculation, not the adapter's training objective.

## 5. Split structure and why it is not a 14-patient evaluation

The data are explicitly reduced to one sample:

```text
SAMPLE_ID = "sample_1"
ds["patient"] = SAMPLE_ID
```

The first split groups matched tiles into spatial blocks of `224 * 10 = 2240` coordinate units, shuffles blocks with seed 42, and assigns approximately 70%/15%/15% to train/validation/test. The recorded counts are 3,417 train, 647 validation, and 832 test tiles out of 4,896 matched tiles.

The 832 test tiles are then randomly divided into 666 development tiles and 166 final-test tiles. Within the 666, Optuna uses 583 inner-train and 83 inner-validation tiles. After tuning, the model is retrained on all 666 development tiles and evaluated on the untouched 166.

A second experiment uses the same 832 initial test tiles, holds out the rightmost approximately 20% by an `x >= 80th percentile` rule, and trains on 665 tiles versus 167 spatial-test tiles. This is a spatial-region holdout, not a patient holdout.

This is not a 14-patient evaluation because:

- only `sample_1` is selected;
- the patient label is assigned programmatically to every DeepSpot row rather than read as a multi-patient grouping variable;
- the bulk vector is one constant vector repeated for every tile;
- no patient-level train/test partition, leave-one-patient-out procedure, per-patient metric, or cross-patient aggregation exists;
- the reported tile counts (4,896, 832, 666, 166, 665, and 167) are spatial observations from one specimen, not 14 independent patients.

Consequently, the reported held-out performance measures within-sample spatial generalization and, in the random case, nearby/related tile generalization. It cannot establish performance on unseen patients or demonstrate robustness across 14 patient-specific bulk and Xenium distributions.

## 6. Migration issues

- **Kaggle paths:** all three inputs are absolute `/kaggle/input/...` paths. Fallback discovery recursively searches `/kaggle/**/*...`; this is environment-specific and can select an arbitrary first match. Outputs are hard-coded under `/kaggle/working/`, including CSV, NPY, PT, and PNG files, plus `/kaggle/working/spatial_diagnostic_plots`.
- **CPU-only behavior:** `DEVICE = 'cpu'` is hard-coded. The code prints device but does not make device selection configurable. The `torch.cuda.is_available()` cleanup branch does not change that setting.
- **Package mutation:** the Optuna section detects a missing package and runs `pip install -q optuna` through `subprocess`. This is unsuitable for a reproducible project pipeline and violates an offline/no-install execution contract.
- **Notebook-only execution:** `# %%` cell markers, `display(...)`, repeated `plt.show()`, and `show_progress_bar=True` assume an interactive notebook. The code is stateful: later cells require globals such as `P_test`, `Y_test`, `genes`, `b`, `te`, `ds_sub`, `spatial_results`, and `pred_test` from prior cells.
- **Execution-order fragility:** cells redefine imports, losses, constants, and model classes; conditional checks such as `if "selected_idx" not in globals()` alter behavior depending on prior execution. There is no callable data-preparation, train, evaluate, or save interface.
- **Memory and scale:** the full transcript table is loaded and transformed in memory; grouping/pivoting all retained transcripts and full-batch PyTorch training are not designed for a multi-patient production dataset.
- **Configuration leakage:** sample ID, registration matrix, pixel sizes, image dimensions, tile size, quality threshold, seed, hyperparameters, and output names are literals rather than a versioned configuration.
- **Alignment validation:** the transform is assumed, inverted without a registration-quality check, and applied with implicit axis/sign conventions. No landmark residual, overlap visualization, or per-sample transform validation is performed.
- **Data semantics:** the fallback for a one-patient bulk file can silently substitute a patient; missing bulk genes are silently median-filled; controls are identified by a name-prefix fallback; and the 305-gene assertion is permissive (`>20`) rather than enforcing a registered gene vocabulary.
- **Evaluation ambiguity:** the “random 80% Xenium loss test” takes 80% of an already held-out tile set, while a later Optuna experiment calls that subset development data. The source also reports diagnostic/bootstrap results from the spatial holdout. These outputs need an unambiguous evaluation manifest and fixed metric definitions.
- **Reproducibility gaps:** seeds are set for Python/NumPy/PyTorch, but there is no explicit deterministic PyTorch configuration, data/version manifest, transform provenance, or persisted preprocessing schema in the basic workflow.
- **Model-checkpoint coupling:** the saved checkpoint includes weights and some arrays, but not the registration metadata, full input schema, control-filter rules, bulk fallback decisions, or a stable model class/configuration identifier.

## 7. Reuse versus redesign

### Safe to reuse as a documented baseline or starting component

- The three-source data contract, after replacing paths with explicit configuration and validating schemas.
- The transcript QC concepts: quality-value threshold, `is_gene` filtering, control-prefix fallback, byte decoding, coordinate bounds, and transcript-to-spot aggregation.
- The coordinate math and 224-pixel tile assignment as a candidate transform, provided the matrix and dimensions are independently verified for each sample.
- The sorted shared-gene intersection and explicit gene ordering, provided the project records a versioned gene manifest and does not silently accept a changed intersection.
- Library-size normalization to CP10K followed by `log1p`, and training-subset-only z-score fitting, as an auditable baseline preprocessing option.
- The bounded per-tile scale adapter with gene intercepts and gene bulk gains as a simple interpretable calibration baseline.
- MSE, gene-wise spatial Pearson, per-tile gene-rank Spearman, aggregate gene-level metrics, and paired same-gene comparisons as candidate reporting metrics.
- The pairwise rank and scale penalties as optional ablations whose weights and sampling policy are explicitly configured.
- Seeded experiments, gradient clipping, AdamW, and the recorded Optuna search space as reproducibility references—not as a substitute for a production experiment manager.

### Must be redesigned before claiming a project-level or 14-patient result

- **Data ingestion:** replace Kaggle discovery and notebook globals with validated, configurable per-patient manifests; stream or chunk transcript processing; record file checksums and schema versions.
- **Registration:** externalize per-sample H&E/Xenium transforms and dimensions, validate them with landmarks/overlays and residual thresholds, and handle axis conventions and registration failures explicitly.
- **Patient modeling:** load all patients, keep patient identity from source metadata, support one bulk vector per patient, and define how patient-specific genes, missing genes, and batch effects are handled.
- **Splitting:** use patient-level train/validation/test partitions (or leave-one-patient-out/cross-validation), with spatial blocking nested within training patients when needed. Never use tiles from a test patient for tuning or normalization.
- **Normalization:** fit and persist all statistics on the correct training patients only; define whether bulk normalization is within patient or cohort-level; replace silent median filling with an explicit missingness policy and sensitivity analysis.
- **Architecture:** treat the current same-gene bulk-gated scale model as a baseline. For cross-patient prediction, redesign or justify patient conditioning, batch correction, uncertainty, and the ability to model cross-gene relationships; compare against bulk-only and DeepSpot-only baselines.
- **Training objective:** make batch composition explicit for correlation and rank losses, evaluate whether full-batch correlation is appropriate, tune loss weights only within training folds, and report ablations for MSE/Pearson/rank/scale terms.
- **Evaluation:** pre-register primary metrics, report per-patient distributions and confidence intervals, separate patient-level and tile-level summaries, and preserve a genuinely untouched patient test set. Do not select top genes on the same data used for final performance claims.
- **Packaging and operations:** remove runtime installation and plotting side effects; expose library/CLI entry points; make CPU/GPU selection configurable; add tests for schema, alignment, gene order, leakage, and checkpoint round trips; persist complete preprocessing and model metadata.

