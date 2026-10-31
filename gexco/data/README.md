# Data preparation

Run commands from the repository root after [installation](../../README.md#installation).
Use [preparation/default.yaml](../configs/preparation/default.yaml) for the settings below.

## Data splits

The released lists contain one ID per line:

| Cohort | IDs | File |
|---|---:|---|
| CT-RATE train | 32,941 volumes | [ctrate_train.txt](../../data_splits/ctrate_train.txt) |
| CT-RATE test | 2,032 volumes | [ctrate_test.txt](../../data_splits/ctrate_test.txt) |
| LIDC acquired-radiograph cohort | 155 cases | [lidc_xray_155.txt](../../data_splits/lidc_xray_155.txt) |

CT-RATE IDs identify individual reconstructions, such as `train_1_a_2`;
LIDC IDs identify subjects, such as `LIDC-IDRI-0001`. Obtain images from the
dataset providers under their data-use terms and map these IDs to your local files.

## Data and geometry

| Parameter | Default |
|---|---|
| CT grid | 128³ at 3 mm; HU clipped to [-1000, 1000] |
| Source–detector / source–isocentre distance | 1500 / 972 mm |
| Nominal detector | 512² at 1.2 mm |
| Stored DRR sampling | Direct 128² at 4.8 mm |
| Angular grid | -90° to +90° in 15° steps; 13 views |
| Biplanar input | Frontal 0° and lateral 90° |
| DiffDRR pose | AP orientation, ZXY Euler angles, yaw in the first component |
| Density / detector convention | `diffdrr` / `standard` |

A preparation manifest is JSON or JSONL with one record per CT:

```json
{
  "samples": [{
    "case_id": "case_0001",
    "patient_id": "patient_0001",
    "volume": "data/ct128/case_0001.nii.gz",
    "spacing_mm": [3, 3, 3]
  }]
}
```

## CT preprocessing

Prepare RAS-oriented HU NIfTIs with modality decoding applied upstream:

```bash
gexco preprocess-ct --input data/native/case_0001.nii.gz --output data/ct128/case_0001.nii.gz
```

The helper clips HU, resamples toward 3 mm, resizes to 128³, and writes a
centered 3 mm affine plus geometry sidecar. The final resize scales anatomy when the
native field of view differs from 384 mm. Array-axis ordering is retained
through latent extraction and reconstruction.

### DRR generation and HDF5 export

```bash
gexco make-drr --input data/ct128/case_0001.nii.gz --output-dir data/drr

gexco export-hdf5 --manifest data/ct_manifest.json --output-dir data/drr_h5
```

`make-drr` writes raw float32 `<case>_F.tiff`, `<case>_L.tiff` and geometry metadata.
`export-hdf5` writes `shard_*.h5` with `/views/128`, separate `/index/case_id`
and `/index/patient_id`, `/index/ct_path` and `/index/complete`, together with density, orientation,
geometry and angle attributes. The frontal/lateral angle indices are 6 and 12.
Supply `patient_id` in the CT manifest when multiple volumes belong to one patient;
it is preserved when rebuilding manifests from HDF5. Unknown patient IDs are stored
as empty strings. Older banks with case IDs in `/index/patient_id` remain readable.
The stored effective pixel spacing is derived from the configured field of view
and output resolution; `delx_128` records the equivalent spacing at 128 pixels.

Use `world_model_manifest.json` for all 13 views and `biplanar_manifest.json`
for the 0°/90° pair. `hdf5.shard_size` controls shard size. Completed shards
are not overwritten: use a fresh output directory for each export. Loaders
validate density/orientation and read scaling from the HDF5 metadata.

## Acquired radiographs

Prepare frontal/lateral DICOM pairs with the paper's unregistered preprocessing:

```bash
gexco prepare-xray --frontal data/radiographs/case_0001_F.dcm \
  --lateral data/radiographs/case_0001_L.dcm --case-id case_0001 \
  --output-dir data/radiographs_prepared
```

For a cohort, replace the two input arguments with `--manifest pairs.json`:

```json
{"samples": [{"case_id": "case_0001", "frontal": "data/case_0001_F.dcm", "lateral": "data/case_0001_L.dcm"}]}
```

Views are assigned explicitly; supply upright images in frontal/lateral order.
The pipeline applies the modality LUT/rescale, inverts MONOCHROME1, resamples
to **0.7776 mm/pixel at isocentre**, centers the radiograph body bounding box,
and crops/pads to 512². Its body mask uses full-image p1/p99 scaling,
threshold 0.05, 5×5 closing, the largest connected component and hole filling.
Body p1/p99 intensities map to the fixed training-DRR p1/p99 values before
normalization to [0,1]; 128² inputs are obtained by 4×4 area averaging.
VOI/window transforms are omitted. No CT or CT-guided registration is required.

[radiographs/default.yaml](../configs/radiographs/default.yaml) contains the
paper calibration. Detector spacing comes from `ImagerPixelSpacing`, then a
matching manufacturer/model/image-size donor group, then the recorded cohort
median (0.194444 mm). Magnification is estimated as source-detector distance
divided by source-support distance minus half the assumed thorax depth
(250 mm frontal, 306 mm lateral). Missing distance tags use 1800 mm and a
50 mm source-detector to source-support gap. This approximation assumes
contact with the support and does not correct residual perspective differences.
Copy the YAML and pass `--config` to use other acquisition calibration.

Outputs are float32 TIFFs in `512/<case>/` and `128/<case>/`, an inference
`manifest.json`, and `preprocessing.json` with source hashes, intensity/scale
decisions and missing-tag flags, including tier-c views and affected pairs.
Use a fresh or empty output directory. Run inference with
`--input-kind normalized --manifest data/radiographs_prepared/manifest.json`,
or [pass the DICOM pair directly](../reconstruction/README.md#radiograph-inputs).

## Download the frozen MAISI VAE

Use `AutoencoderKlMaisi`, `autoencoder_epoch273.pt`, from the
`maisi_ct_generative` bundle **v0.3.6**:

```bash
python scripts/download_checkpoints.py --maisi --output-dir checkpoints
```

The downloader verifies the [official asset](https://github.com/Project-MONAI/model-zoo/blob/dev/models/maisi_ct_generative/large_files.yml)
SHA-256: `1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3`.
MAISI weights remain subject to [NVIDIA's license](../../README.md#license).

## Extract CT latents

```bash
gexco extract-latents --manifest data/drr_h5/biplanar_manifest.json \
  --vae-checkpoint checkpoints/autoencoder_epoch273.pt --output-dir data/latents
```

Extraction uses the frozen VAE posterior with a per-case seed. Outputs are
unscaled `4 × 32 × 32 × 32` channel-first `.npy` latents and
`data/latents/manifest.json`, preserving CT geometry and view references.
Training applies scale factor **0.941389799118042**; inference removes it before
decoding. CUDA VAE execution uses FP16 and CPU execution uses FP32.

Continue with [world-model pretraining](../pretraining/README.md) and
[reconstruction training](../reconstruction/README.md#training).
