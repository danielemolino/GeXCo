"""Recreate the supervised CT-Net evaluation model from CT128 HU volumes."""

import argparse
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from gexco.config import load_config, default_config
from gexco.evaluation.ctnet import CTNetModel
from gexco.evaluation.classification import compute_multilabel_metrics
from gexco.evaluation.run_suite import _index
from gexco.data.io import case_id
from gexco.utils.tensors import seed_everything
from gexco.utils.checkpoint import atomic_save


class LabeledCTs(Dataset):
    def __init__(self, root, csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.labels = [c for c in reader.fieldnames if c != "VolumeName"]
            rows = list(reader)
        index = _index(root)
        self.records = [(index[case_id(row["VolumeName"])], row) for row in rows if case_id(row["VolumeName"]) in index]
        if not self.records:
            raise ValueError("No labeled CTs")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, row = self.records[i]
        image = nib.load(str(path)).get_fdata(dtype=np.float32)
        if image.shape != (128, 128, 128):
            raise ValueError("Classifier requires CT128")
        image = ((np.clip(image, -1000, 1000) + 1000) / 2000)[:126].reshape(42, 3, 128, 128)
        return torch.from_numpy(image.copy()), torch.tensor([float(row[c]) for c in self.labels])


def train(config):
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = LabeledCTs(config["train_volumes"], config["train_labels"])
    valid_ds = LabeledCTs(config["validation_volumes"], config["validation_labels"])
    if train_ds.labels != valid_ds.labels:
        raise ValueError("Class label order differs")
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )
    valid_loader = DataLoader(valid_ds, batch_size=config["batch_size"], num_workers=config["num_workers"])
    model = CTNetModel(len(train_ds.labels))
    if config["imagenet_init"]:
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.features.load_state_dict(torch.nn.Sequential(*list(resnet.children())[:-2]).state_dict(), strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        for image, target in train_loader:
            logits = model(image.to(device))
            loss = criterion(logits, target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite classifier loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        truth = []
        scores = []
        model.eval()
        with torch.inference_mode():
            for image, target in valid_loader:
                truth.extend(target.numpy())
                scores.extend(torch.sigmoid(model(image.to(device))).cpu().numpy())
        metrics = compute_multilabel_metrics(np.asarray(truth), np.asarray(scores))
        score = metrics["mean_auroc"]
        payload = {"model": model.state_dict(), "label_columns": train_ds.labels, "config": config, "epoch": epoch + 1}
        atomic_save(payload, out / "last.pt")
        if score > best or not (out / "best.pt").exists():
            best = score
            atomic_save(payload, out / "best.pt")
        row = {"epoch": epoch + 1, "loss": total / len(train_loader), **metrics}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return out / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config("classifier"))
    args = parser.parse_args()
    print(train(load_config(args.config)))


if __name__ == "__main__":
    main()
