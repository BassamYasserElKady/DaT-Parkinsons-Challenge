# Multi-sample spatial-transcriptomics calibration

This repository contains a clean, parameterized skeleton for calibrating DeepSpot-M tile predictions against matched Xenium measurements across 12 samples:

- 10 `development` samples for fitting, grouped validation, and model selection.
- 4 `final_test` samples held untouched until the final report.

The first model uses only DeepSpot-M tile predictions as input. Bulk RNA-seq, microarray, and scRNA-seq data from independent patients are deliberately not conditioning inputs and are not part of this first pipeline.

The legacy notebook remains unchanged at `notebooks/legacy/`. It is reference material, not an executable dependency.

## Layout

```text
configs/default.yaml                 Parameter defaults
data/sample_manifest.template.csv    Per-sample input template
docs/multisample_design.md           Scientific and leakage-control design
src/spatial_pipeline/                Reusable library code
scripts/prepare_data.py              Validate, align, intersect, and save matrices
scripts/evaluate.py                  Evaluate saved predictions without training
tests/                               Lightweight manifest/split tests
```

## Manifest

Copy `data/sample_manifest.template.csv` to `data/sample_manifest.csv` and fill all 12 rows. Paths are relative to the project root. Each row must contain the sample's DeepSpot-M parquet, Xenium transcript parquet, full and registration-scale H&E dimensions, tile size, Xenium micron-per-pixel scale, and a 3x3 affine transform serialized row-major with semicolon separators.

The manifest must have exactly 8 `development` rows and 4 `final_test` rows. It is the only source of sample-specific geometry and input paths.

## Safe data preparation

The command below validates the manifest, checks input schemas, maps transcripts to DeepSpot-M tiles, computes the development-derived gene vocabulary, applies it to every sample, and writes one compressed matrix file per sample plus development-only normalization statistics.

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --root . \
  --manifest data/sample_manifest.csv \
  --output data/processed
```

This command reads real data but does not train a model or download anything. It must not be run until the manifest points to available local inputs.

## Model and evaluation boundary

`DeepSpotCalibrationAdapter` in `src/spatial_pipeline/adapter.py` is the first trainable model definition. It accepts a normalized `[tiles, genes]` DeepSpot-M matrix and predicts calibrated Xenium values. There is no patient embedding, bulk vector, microarray input, or scRNA-seq input.

The library provides loss terms and metrics, but intentionally contains no training driver. A future trainer must fit normalization on development samples only, perform grouped validation by sample, and keep the four `final_test` samples out of all selection decisions.

Saved predictions can be evaluated with:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --manifest data/sample_manifest.csv \
  --processed-dir data/processed \
  --predictions-dir results/predictions \
  --output results/multisample_metrics.csv
```

Prediction files are expected to be `<sample_id>.npy`, with the same row order and gene order as the corresponding processed matrix. Evaluation reports each sample separately and a pooled result; it does not use the legacy Cell 3 tile split as a final evaluation design.

## Checks without real data

```bash
PYTHONPYCACHEPREFIX=/tmp/spatial_transcriptomics_pycache \
  python -m compileall -q src scripts tests
pytest -q -p no:cacheprovider --basetemp=/tmp/spatial_transcriptomics_pytest
```

The tests exercise only manifest validation and sample-level leakage protection. No data, model, or training run is required.

