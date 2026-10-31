# Any-Anchor world-model pretraining

Run commands from the repository root after [installation](../../README.md#installation).
First [export the 13-view HDF5 bank](../data/README.md#drr-generation-and-hdf5-export).

## Training

Copy [pretraining/default.yaml](../configs/pretraining/default.yaml) and set
`data.manifest` to `world_model_manifest.json`:

```bash
gexco pretrain-wm --config my_pretraining.yaml --output-dir runs/world_model
```

| Setting | Default |
|---|---|
| Encoder | ViT 128², patch 8, width 128, depth 12, 4 heads |
| Encoder projector | 128 → 512 → 128 |
| View predictor | AdaLN, depth 3, delta-angle actions |
| Objective | Shared-encoder prediction MSE + SIGReg |
| Targets / regularization | K=8 / lambda=0.1 |
| Schedule | 100 epochs, batch 64, AdamW, seed 42 |
| Learning rate | 2e-5 → 1e-4 → 1e-6; 10-epoch warmup |
| Weight decay | Cosine schedule, 0.04 → 0.4 |

Pretraining holds out 10% of training patients at seed 42. Supply `patient_id`
for datasets with repeated scans.
Outputs are `last.pt`, holdout-selected `best.pt`, `history.jsonl` and `split.json`.
Selection uses held-out MSE + lambda × SIGReg.

Set `training.resume` to continue from an epoch checkpoint while retaining the
original epoch target, input manifests, batch size and worker count.

## Using the encoder

Set `encoder.checkpoint` to `runs/world_model/best.pt` for
[reconstruction training](../reconstruction/README.md#training).
