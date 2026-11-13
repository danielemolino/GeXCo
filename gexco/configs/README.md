# YAML configuration

Copy a stage YAML, edit its paths and pass `--config my_config.yaml`.
The root `configs` symlink points to this packaged directory.

| Stage | YAML | Instructions |
|---|---|---|
| CT/DRR/HDF5 preparation | [preparation/default.yaml](preparation/default.yaml) | [Data preparation](../data/README.md) |
| Acquired radiographs | [radiographs/default.yaml](radiographs/default.yaml) | [DICOM preprocessing](../data/README.md#acquired-radiographs) |
| World-model pretraining | [pretraining/default.yaml](pretraining/default.yaml) | [Pretraining](../pretraining/README.md) |
| Generator training/inference | [reconstruction/default.yaml](reconstruction/default.yaml) | [Reconstruction](../reconstruction/README.md) |
| Evaluation | [evaluation/default.yaml](evaluation/default.yaml) | [Evaluation](../evaluation/README.md) |
| CT-Net training | [classifier/default.yaml](classifier/default.yaml) | [Classifier recipe](classifier/README.md) |

User paths resolve from the current working directory. The fixed DRR calibration
JSON and `classification.checkpoint: bundled:ctnet` resolve inside the installed
package. Python callers can obtain defaults with `gexco.config.default_config(stage)`.
