# GeXCo package

[Installation and quickstart](../README.md#installation).
Run all commands from the repository root.

## Modules

| Directory | Instructions |
|---|---|
| [data](data/README.md) | Dataset splits, CT preprocessing, DRRs, HDF5 and MAISI latents |
| [pretraining](pretraining/README.md) | Any-Anchor world-model training |
| [reconstruction](reconstruction/README.md) | Generator training and two-view inference |
| [evaluation](evaluation/README.md) | Volume, reprojection, segmentation, CT-Net and FRD metrics |
| [configs](configs/README.md) | YAML defaults and path resolution |
| [weights](weights/README.md) | Checkpoint locations and requirements |
| [utils](utils/README.md) | Seeding, checkpoint writing and patient-level splitting |

## Runtime

The pinned environment uses Python 3.11, PyTorch 2.9.1 and MONAI 1.5.1.
Git and a C/C++ compiler are required for PyRadiomics. Training and inference
use CUDA; [tests and the pipeline smoke](../tests/README.md) run on CPU.

## Expected outputs and measured runtimes

| Stage | Output |
|---|---|
| Preparation | CT128 NIfTIs, frontal/lateral TIFFs, HDF5 banks and manifests |
| Pretraining | World-model checkpoints, history and split IDs |
| Latent extraction | Four-channel 32³ MAISI latents and a training manifest |
| Reconstruction training | Generator checkpoint and training history |
| Inference | One CT128 int16 HU NIfTI per case |
| Evaluation | Per-case and aggregate metrics, segmentation/radiomics caches |

The manuscript reports approximately **1.9 s per reconstruction on one NVIDIA A40**.
Runtime depends on hardware, cohort size and configuration. Segmentation and
radiomics take longer than volume metrics.
