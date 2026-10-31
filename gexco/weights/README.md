# Model checkpoints

| Component | Location |
|---|---|
| CT-Net evaluator | Bundled `ctnet.pt`; checksum and 18-label order in [manifest.json](manifest.json) |
| MAISI VAE | [Official download](../data/README.md#download-the-frozen-maisi-vae) |
| World model | [Pretrained Any-Anchor checkpoint](https://github.com/danielemolino/GeXCo/releases/download/v1.0.0/world_model.pt) |
| GeXCo generator | [Reconstruction checkpoint](https://github.com/danielemolino/GeXCo/releases/download/v1.0.0/gexco.pt), including its finetuned encoder |

Download both models with SHA-256 verification:

```bash
python scripts/download_checkpoints.py --gexco --output-dir checkpoints
```

Add `--maisi` to download the frozen VAE as well. Sizes, URLs and checksums are in
[release.json](release.json); the [Release](https://github.com/danielemolino/GeXCo/releases/tag/v1.0.0)
also provides `SHA256SUMS`. These model files support inference and encoder
initialization; resumable training checkpoints are written by the training commands.

The evaluation configuration resolves CT-Net inside the installed package.
CT-RATE-trained weights retain the data's non-commercial terms; see the
[code and weight licenses](../../README.md#license).
