# Reconstruction training and inference

Run commands from the repository root after [installation](../../README.md#installation).
Prepare a [world-model checkpoint](../pretraining/README.md) and
[MAISI latents](../data/README.md#extract-ct-latents).

## Training

Copy [reconstruction/default.yaml](../configs/reconstruction/default.yaml).
Set `training.manifest` to `data/latents/manifest.json`, `encoder.checkpoint`
to the pretrained encoder, and `training.output_dir` to the run directory:

```bash
gexco train-gen --config my_reconstruction.yaml
```

| Setting | Default |
|---|---|
| Training | 200 epochs, batch 32, FP32 |
| Optimizer | Adam with polynomial LR decay |
| Denoiser/fusion LR | 1e-4 |
| Encoder/projector LR | 1e-5 |
| CFG conditioning dropout | 0.1 |
| Rectified-flow target | `data - noise`, continuous uniform timesteps |

Set `encoder.trainable: false` to freeze the encoder and its BatchNorm state.
The denoiser concatenates the four noisy latent channels with the 128-channel
geometry-conditioned volume. MAISI latents remain fixed during training.
The encoder and denoiser architecture fields specify the fixed paper model;
unsupported changes are rejected before loading weights.

Training writes `unet_3d_last.pt` and `history.jsonl`. Set `training.resume`
to continue from a saved checkpoint.

### Optional training monitoring

Monitoring is disabled by default (`training.validation: false`). To enable it:

```yaml
training:
  validation: true
vae:
  checkpoint: checkpoints/autoencoder_epoch273.pt
```

Each epoch reconstructs the same five samples from `training.manifest`, selected
using the training seed. Their IDs are saved in `validation_samples.json`.
Only mean volumetric **PSNR and SSIM** are recorded under `validation` in
`history.jsonl`, following the [HU metric settings](../evaluation/README.md#psnr-and-ssim).
This is training-set monitoring; it does not select checkpoints.

## Inference

```bash
gexco infer --checkpoint checkpoints/gexco.pt \
  --vae-checkpoint checkpoints/autoencoder_epoch273.pt \
  --frontal data/drr/case_0001_F.tiff --lateral data/drr/case_0001_L.tiff \
  --output-dir predictions --seed 0
```

Supply [local model checkpoints](../weights/README.md). For HDF5 inputs, replace
the frontal/lateral arguments with `--manifest data/drr_h5/biplanar_manifest.json`.
Each record must have exactly the 0°/90° pair.

Inference uses the configuration saved in the checkpoint. For checkpoints without
saved settings, it uses the packaged defaults. An explicit `--config` may change
sampling options and local checkpoint/statistics paths; geometry, normalization
and flow settings must match the saved training configuration. Relocated statistics
are checked against their saved SHA-256 checksum.

Defaults are **30 ODE steps, CFG 1.5 and seed 0**. Override sampling with
`--ode-steps` and `--guidance-scale`. Conditioning is cached; set
`inference.conditioning_cache: false` to recompute it at every step.

The default `per_case_fixed` seed policy is independent of manifest order.
`inference.seed_policy: per_case_index` adds the case index to the seed;
a record-level `seed` takes precedence. Outputs are 128³ int16 HU NIfTIs at
3 mm spacing, named by case ID. A supplied affine is preserved; otherwise
a centered 3 mm affine is used.

### Radiograph inputs

Use `--input-kind acquired` with frontal/lateral **DICOM** radiographs to apply
the [paper preprocessing](../data/README.md#acquired-radiographs) during inference:

```bash
gexco infer --checkpoint checkpoints/gexco.pt \
  --vae-checkpoint checkpoints/autoencoder_epoch273.pt \
  --frontal data/radiographs/case_0001_F.dcm \
  --lateral data/radiographs/case_0001_L.dcm \
  --input-kind acquired --output-dir predictions
```

Both views must be supplied upright in the expected frontal/lateral orientation.
Use `--radiograph-config my_radiographs.yaml` to change metadata calibration.
Alternatively, run `gexco prepare-xray` once and pass its `manifest.json` to
`gexco infer --input-kind normalized`. Prepared [0,1] TIFFs are loaded directly.

Score the exported volumes with the [evaluation suite](../evaluation/README.md).
See [runtime estimates](../README.md#expected-outputs-and-measured-runtimes).
