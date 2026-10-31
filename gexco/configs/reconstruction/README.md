# Reconstruction configuration

Copy [default.yaml](default.yaml) and set `training.manifest`,
`encoder.checkpoint` and `training.output_dir`.
See [training and inference](../../reconstruction/README.md) for defaults,
optional five-sample monitoring and checkpoint inputs.

Encoder and denoiser architecture fields specify the paper model and are validated;
they are not architecture-search options. Inference reads saved checkpoint settings
unless a compatible `--config` override is supplied.
