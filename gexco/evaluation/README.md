# Evaluation

Run commands from the repository root after [installation](../../README.md#installation).
The suite accepts folders of HU NIfTIs from any reconstruction method.
Files are matched recursively by case ID; prediction and reference grids must match.
For repeated filenames in nested folders, supply a `--manifest` with
`case_id`, `prediction` and `reference` for each pair.

```bash
gexco evaluate --predictions predictions --targets data/reference_ct128 \
  --metrics volume reprojection segmentation frd classification \
  --labels data/labels.csv \
  --output-dir evaluation
```

Omit `--metrics` for the full suite, or use `--metrics volume` for a CPU smoke.
`--max-cases N` selects the first N sorted case IDs for all requested metrics.
Results are saved in `metrics.json`, with per-case values and aggregate scores.
Segmentation and radiomics caches are keyed by source identity.

| Metric | Settings |
|---|---|
| CT PSNR / SSIM | Clip both volumes to [-1000, 1000] HU; range 2000 HU |
| Reprojection PSNR / SSIM | Frontal/lateral DRRs; fixed normalization; range 1 |
| Anatomy | TotalSegmentator 2.18.0; 29 non-rib structures; Dice, HD95, NSD@3 mm |
| Classification | CT-Net AUROC and AP |
| FRD | Official `frd-score==1.0.1`, version v1 |

## PSNR and SSIM

Volume metrics use the full CT and fixed **2000 HU** range. SSIM uses MONAI's
3D Gaussian windows of size 11. Identical pairs have PSNR `Infinity` and SSIM 1.

Reprojection clips both CTs to [-1000, 1000] HU and renders the frontal/lateral
views with the [preparation geometry and density settings](../data/README.md#data-and-geometry).
Both views use fixed training p1/p99 normalization and **data range 1** in
normalized projection units. Scores are averaged over the two views.
Reference projections are generated from the reference CT.

## TotalSegmentator anatomy

Reference and prediction are segmented independently with **TotalSegmentator
2.18.0**, task `total`, full resolution, multilabel output, seed 0 and one
resampling/saving worker. First use downloads the model weights.

The **29 non-rib structures** are five lung lobes, oesophagus, trachea,
vertebrae L1 and T1–T12, heart, aorta, pulmonary vein, left/right scapulae,
left/right clavicles, spinal cord and sternum. IDs are listed in
[`segmentation.py`](segmentation.py).

Metrics are averaged across structures within each volume, then across volumes.
HD95 is the maximum of the two directed 95th-percentile surface distances,
in millimetres. NSD uses subvoxel surface areas with inclusive **3 mm** tolerance.
Missing structures follow the same rules for every method:

| Reference | Prediction | Dice | NSD | HD95 |
|---|---|---:|---:|---|
| Present | Absent | 0 | 0 | **384√3 = 665.1075101064489 mm**, included in the mean |
| Absent | Absent | Excluded | Excluded | Excluded |
| Absent | Present | 0 | 0 | **384√3 = 665.1075101064489 mm**, included in the mean |

When exactly one mask is empty, HD95 is assigned the diagonal of the 384 mm
CT128 voxel support as a finite penalty. This convention applies symmetrically
to missing and extra structures; structures absent from both masks are excluded.

JSON reports encode non-finite numbers as the strings `"Infinity"`, `"-Infinity"`
and `"NaN"`. For example, identical CTs have PSNR `"Infinity"`. Excluded or missing
metrics remain `null`; numerical computations use the original floating-point values.

## CT-Net

The [bundled CT-Net checkpoint](../weights/README.md) loads by default.
`--classification-checkpoint` accepts a compatible alternative.
The model uses ResNet-18 slice features, 3D reduction convolutions and 14×14
spatial pooling. CTs are clipped to [-1000, 1000] HU and mapped to [0,1];
the first 126 slices along array axis 0 form 42 triplets.

The labels CSV must contain `VolumeName` or `case_id` and named binary columns.
Checkpoint `label_columns` determines class order; labels are matched by name.
RadChestCT uses the paper's 16-label common ontology, supplied explicitly in the CSV.
AUROC assigns half credit to ties; AP uses sklearn's threshold-based definition.
Undefined per-class metrics are excluded from the macro mean.

For optional evaluator training, see the [classifier configuration](../configs/classifier/README.md).

## Official FRD

The suite uses [official FRD](https://github.com/RichardObi/frd-score),
`frd-score==1.0.1` with `frd_version=v1`, for feature extraction, normalization
and distance calculation. At least two matched cases are required.

Body masks use HU > -900, hole filling, one binary opening and connected
components of at least 5000 voxels, falling back to the largest component.
Image/mask pairs are collected in a source-keyed cache before extraction.
PyRadiomics is pinned in [requirements.txt](../../requirements.txt).

See [evaluation/default.yaml](../configs/evaluation/default.yaml) for all options.
