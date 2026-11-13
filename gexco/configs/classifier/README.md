# CT-Net training configuration

Evaluator retraining is optional. Copy [default.yaml](default.yaml) and set
`train_volumes`, `train_labels`, `validation_volumes` and `validation_labels`
to real CTs and binary label CSVs. Use separate training and validation patients.

```bash
gexco train-classifier --config my_classifier.yaml
```

Defaults: ImageNet-initialized ResNet-18, AdamW LR 1e-4, 100 epochs, batch 32,
seed 42. The trainer saves `best.pt` using validation mean AUROC.
See [CT-Net evaluation](../../evaluation/README.md#ct-net) for preprocessing
and label matching.
