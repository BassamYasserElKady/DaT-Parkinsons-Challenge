# 10x dataset inventory

Inventory root: `/home/ubuntu/spatial-transcriptomics/data/raw/10x`

This is a read-only inventory based on filesystem names and sizes, TIFF headers, the small alignment CSVs, `experiment.xenium` metadata members, and ZIP central directories. No ZIP was extracted, no raw file was modified, and no model training was run.

## Executive summary

- The dataset contains 12 populated samples: `S1`, `S2`, `S3`, and `S4`, each at positions `Bot`, `Mid`, and `Top`.
- Every populated sample has exactly one H&E image, one Xenium archive, one cell-group CSV, and one H&E-alignment CSV.
- Every Xenium archive has 12 members: seven named files, one `morphology_focus/` directory, and four focus-channel TIFFs.
- The required expression member is `cell_feature_matrix.zarr.zip`; the transcript-coordinate member is `transcripts.zarr.zip`. `cells.zarr.zip` provides cell-level data and `experiment.xenium` provides metadata.
- On-disk raw storage is exactly **96,694,798,870 bytes** (**96.695 GB decimal; 90.054 GiB**).
- No checksum sidecar or checksum manifest was found by filename (`.md5`, `.sha*`, or `checksum*`). Hashes were not computed.
- No standalone DeepSpot-M prediction file is present under this raw root. Xenium expression and coordinate data are nested inside the outer `*_xe_outs.zip` archives.
- No train/test split was supplied by the inventory request. `data/sample_manifest.csv` therefore marks all rows `unassigned`; it does not assume the earlier 14-sample/10-development/4-test design.

## Sample inventory

All paths below are relative to the project root in `data/sample_manifest.csv`.

| Sample | Section/position | H&E image (bytes; dimensions) | Xenium archive (bytes) | Cell groups CSV (bytes) | Alignment CSV (bytes) |
|---|---|---:|---:|---:|---:|
| `S1_Bot` | `section_1 / Bot` | `Human_Breast_Biomarkers_S1_Bot_he_image.ome.tif`<br>4,173,029,209; 29,458 x 45,088 | `Human_Breast_Biomarkers_S1_Bot_xe_outs.zip`<br>5,942,405,996 | `Human_Breast_Biomarkers_S1_Bot_cell_groups.csv`<br>7,586,544 | `Human_Breast_Biomarkers_S1_Bot_he_imagealignment.csv`<br>124 |
| `S1_Mid` | `section_1 / Mid` | `Human_Breast_Biomarkers_S1_Mid_he_image.ome.tif`<br>3,187,878,930; 31,283 x 32,331 | `Human_Breast_Biomarkers_S1_Mid_xe_outs.zip`<br>6,546,794,785 | `Human_Breast_Biomarkers_S1_Mid_cell_groups.csv`<br>10,166,162 | `Human_Breast_Biomarkers_S1_Mid_he_imagealignment.csv`<br>125 |
| `S1_Top` | `section_1 / Top` | `Human_Breast_Biomarkers_S1_Top_he_image.ome.tif`<br>2,729,723,481; 20,515 x 43,995 | `Human_Breast_Biomarkers_S1_Top_xe_outs.zip`<br>3,509,308,745 | `Human_Breast_Biomarkers_S1_Top_cell_groups.csv`<br>6,155,398 | `Human_Breast_Biomarkers_S1_Top_he_imagealignment.csv`<br>126 |
| `S2_Bot` | `section_2 / Bot` | `Human_Breast_Biomarkers_S2_Bot_he_image.ome.tif`<br>4,856,200,877; 33,113 x 44,726 | `Human_Breast_Biomarkers_S2_Bot_xe_outs.zip`<br>6,218,590,192 | `Human_Breast_Biomarkers_S2_Bot_cell_groups.csv`<br>9,721,042 | `Human_Breast_Biomarkers_S2_Bot_he_imagealignment.csv`<br>129 |
| `S2_Mid` | `section_2 / Mid` | `Human_Breast_Biomarkers_S2_Mid_he_image.ome.tif`<br>3,366,324,272; 27,235 x 38,540 | `Human_Breast_Biomarkers_S2_Mid_xe_outs.zip`<br>4,453,854,396 | `Human_Breast_Biomarkers_S2_Mid_cell_groups.csv`<br>9,584,460 | `Human_Breast_Biomarkers_S2_Mid_he_imagealignment.csv`<br>125 |
| `S2_Top` | `section_2 / Top` | `Human_Breast_Biomarkers_S2_Top_he_image.ome.tif`<br>3,142,703,184; 27,943 x 35,801 | `Human_Breast_Biomarkers_S2_Top_xe_outs.zip`<br>3,602,365,131 | `Human_Breast_Biomarkers_S2_Top_cell_groups.csv`<br>9,243,272 | `Human_Breast_Biomarkers_S2_Top_he_imagealignment.csv`<br>124 |
| `S3_Bot` | `section_3 / Bot` | `Human_Breast_Biomarkers_S3_Bot_he_image.ome.tif`<br>1,312,140,829; 14,853 x 31,607 | `Human_Breast_Biomarkers_S3_Bot_xe_outs.zip`<br>2,113,544,030 | `Human_Breast_Biomarkers_S3_Bot_cell_groups.csv`<br>4,097,219 | `Human_Breast_Biomarkers_S3_Bot_he_imagealignment.csv`<br>125 |
| `S3_Mid` | `section_3 / Mid` | `Human_Breast_Biomarkers_S3_Mid_he_image.ome.tif`<br>3,823,793,193; 32,654 x 35,502 | `Human_Breast_Biomarkers_S3_Mid_xe_outs.zip`<br>7,693,585,320 | `Human_Breast_Biomarkers_S3_Mid_cell_groups.csv`<br>10,341,726 | `Human_Breast_Biomarkers_S3_Mid_he_imagealignment.csv`<br>124 |
| `S3_Top` | `section_3 / Top` | `Human_Breast_Biomarkers_S3_Top_he_image.ome.tif`<br>4,776,423,462; 36,559 x 48,352 | `Human_Breast_Biomarkers_S3_Top_xe_outs.zip`<br>4,562,887,053 | `Human_Breast_Biomarkers_S3_Top_cell_groups.csv`<br>10,418,966 | `Human_Breast_Biomarkers_S3_Top_he_imagealignment.csv`<br>125 |
| `S4_Bot` | `section_4 / Bot` | `Human_Breast_Biomarkers_S4_Bot_he_image.ome.tif`<br>2,215,388,253; 21,607 x 30,322 | `Human_Breast_Biomarkers_S4_Bot_xe_outs.zip`<br>4,009,783,451 | `Human_Breast_Biomarkers_S4_Bot_cell_groups.csv`<br>5,523,438 | `Human_Breast_Biomarkers_S4_Bot_he_imagealignment.csv`<br>122 |
| `S4_Mid` | `section_4 / Mid` | `Human_Breast_Biomarkers_S4_Mid_he_image.ome.tif`<br>2,744,244,311; 22,892 x 39,456 | `Human_Breast_Biomarkers_S4_Mid_xe_outs.zip`<br>5,227,264,262 | `Human_Breast_Biomarkers_S4_Mid_cell_groups.csv`<br>8,813,092 | `Human_Breast_Biomarkers_S4_Mid_he_imagealignment.csv`<br>125 |
| `S4_Top` | `section_4 / Top` | `Human_Breast_Biomarkers_S4_Top_he_image.ome.tif`<br>2,979,124,110; 26,717 x 46,370 | `Human_Breast_Biomarkers_S4_Top_xe_outs.zip`<br>3,410,455,660 | `Human_Breast_Biomarkers_S4_Top_cell_groups.csv`<br>5,332,921 | `Human_Breast_Biomarkers_S4_Top_he_imagealignment.csv`<br>124 |

H&E metadata from the TIFF first image directory: all 12 have three samples/channels and 1024 x 1024 storage tiles. Eleven are BigTIFF; `S3_Bot` is classic TIFF. Dimensions above are width x height in pixels.

## Xenium archive contents

All 12 outer archives have exactly these 12 member paths:

```text
analysis.zarr.zip
experiment.xenium
analysis_summary.html
cell_feature_matrix.zarr.zip
transcripts.zarr.zip
cells.zarr.zip
gene_panel.json
morphology_focus/
morphology_focus/ch0000_dapi.ome.tif
morphology_focus/ch0001_atp1a1_cd45_e-cadherin.ome.tif
morphology_focus/ch0002_18s.ome.tif
morphology_focus/ch0003_alphasma_vimentin.ome.tif
```

The expression/coordinate files required for downstream spatial expression work are:

- `cell_feature_matrix.zarr.zip`: Xenium cell-by-feature expression matrix.
- `transcripts.zarr.zip`: transcript-level records, including the coordinates needed for transcript-to-tile alignment.
- `cells.zarr.zip`: cell-level records and geometry; useful when aggregating by segmented cells or validating boundaries.
- `experiment.xenium`: experiment metadata. Across all 12 samples it reports Xenium software `xenium-4.0.1.1` and `pixel_size = 0.2125`.
- `gene_panel.json`: panel/gene metadata.

The remaining archive members are analysis output and morphology-focus channels. The four focus TIFF names are identical across all samples.

The `experiment.xenium` metadata references `morphology.ome.tif`, but that member is absent from every outer archive. The corresponding full H&E image exists as a separate sibling file named `*_he_image.ome.tif` for every sample. This should be treated as an external-file linkage that must be made explicit in downstream code, not as evidence that `morphology.ome.tif` is present inside the archive.

The cell-group CSV header is identical for all samples:

```text
cell_id,group
```

Each alignment CSV has no header and three rows of three numeric values. The last row is `0,0,1` for every sample. The exact row-major values are recorded in `data/sample_manifest.csv` under `alignment_affine_row_major`.

## Xenium metadata summary

| Sample | Region name | Total transcripts | High-quality transcripts | Cells |
|---|---:|---:|---:|---:|
| `S1_Bot` | `S1_Bot` | 142,039,943 | 117,652,311 | 254,480 |
| `S1_Mid` | `S1_Mid` | 185,957,529 | 151,961,258 | 375,529 |
| `S1_Top` | `S1_Top` | 64,433,598 | 55,283,583 | 209,467 |
| `S2_Bot` | `S2_Bot` | 108,427,269 | 92,260,459 | 358,519 |
| `S2_Mid` | `S2_Mid` | 74,131,254 | 65,062,084 | 380,739 |
| `S2_Top` | `S2_Top` | 45,794,672 | 41,712,807 | 301,854 |
| `S3_Bot` | `S3_Bot` | 43,579,291 | 37,555,926 | 150,145 |
| `S3_Mid` | `S3_Mid` | 211,631,286 | 177,879,847 | 373,150 |
| `S3_Top` | `S3_Top` | 65,513,943 | 57,809,585 | 308,531 |
| `S4_Bot` | `S4_Bot` | 90,354,760 | 80,016,546 | 201,955 |
| `S4_Mid` | `S4_Mid` | 132,251,152 | 111,763,697 | 335,709 |
| `S4_Top` | `S4_Top` | 66,064,413 | 55,927,783 | 192,898 |

## Storage

The exact sum of all regular files under the raw root is **96,694,798,870 bytes**. `du -sb` reports the same total.

| Sample | Total bytes |
|---|---:|
| `S1_Bot` | 10,123,021,873 |
| `S1_Mid` | 9,744,840,002 |
| `S1_Top` | 6,245,187,750 |
| `S2_Bot` | 11,084,512,240 |
| `S2_Mid` | 7,829,763,253 |
| `S2_Top` | 6,754,311,711 |
| `S3_Bot` | 3,429,782,203 |
| `S3_Mid` | 11,527,720,363 |
| `S3_Top` | 9,349,729,606 |
| `S4_Bot` | 6,230,695,264 |
| `S4_Mid` | 7,980,321,790 |
| `S4_Top` | 6,394,912,815 |
| **Total** | **96,694,798,870** |

## Checksums

No checksum files or manifests were found under the raw root by filename search for `.md5`, `.md5sum`, `.sha1`, `.sha256`, `checksum`, or `checksums`. No hashes were computed during this inventory, so the inventory does not provide cryptographic verification of payload integrity.

## Missing and inconsistent files

### Complete per-sample set

No populated sample is missing any of the four requested sibling assets. There are 48 expected sibling files total: 12 H&E images, 12 Xenium archives, 12 cell-group CSVs, and 12 alignment CSVs.

All 12 archives contain the required expression, transcript-coordinate, cell, experiment, and gene-panel members. All 12 also contain four morphology-focus TIFFs.

### Naming/layout issues

- The populated directories use `section_1_bot`, `section_1_mid`, `section_1_top`, through `section_4_top`.
- Empty alias directories also exist for `section_1_bottom`, `section_1_middle`, and the corresponding `section_2`–`section_4` names. They contain no sample files and were not counted as samples.
- `experiment.xenium` references `morphology.ome.tif`, which is absent inside every outer archive; the sibling `*_he_image.ome.tif` is the available H&E asset.
- `S3_Bot` is a classic TIFF while the other 11 H&E images are BigTIFF. It has valid readable TIFF header dimensions but should be covered by a format compatibility test.
- Alignment CSVs have no header. Their values are captured verbatim in the manifest, but transform direction and coordinate convention still require downstream validation against landmarks.
- The raw root has no DeepSpot-M prediction parquet files and no standalone `transcripts.parquet`. A future preparation step must explicitly read the nested archive members or create a separate derived-data manifest; it must not pretend these files already exist.
- No split assignment was discoverable from this raw layout. The manifest uses `unassigned` for all 12 samples until a study-level split is specified.

