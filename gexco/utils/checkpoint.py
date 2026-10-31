"""Atomic state saving and patient-level holdout selection."""

from pathlib import Path
import os
import re
import random
import torch


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def split_training_records(records, fraction=0.1, seed=42):
    def patient(row):
        if "patient_id" in row:
            return str(row["patient_id"])
        match = re.match(r"(?:train|valid)_(\d+)(?:_|$)", str(row["case_id"]))
        return match.group(1) if match else str(row["case_id"])

    ids = sorted({patient(row) for row in records})
    if len(ids) < 2:
        raise ValueError("Pretraining needs at least two patients for train/validation holdout")
    random.Random(seed).shuffle(ids)
    heldout = set(ids[: max(1, round(len(ids) * fraction))])
    return [row for row in records if patient(row) not in heldout], [row for row in records if patient(row) in heldout]
