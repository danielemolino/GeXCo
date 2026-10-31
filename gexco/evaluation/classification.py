"""Multi-abnormality AUROC/AP and checkpoint-exact CT-Net evaluation."""

from __future__ import annotations

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _numpy(value: ArrayLike) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def binary_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute binary AUROC using pairwise ranking.

    Args:
        labels: Binary ground-truth labels.
        scores: Continuous prediction scores.

    Returns:
        Area under the ROC curve, or NaN when one class is absent.
    """
    truth = _numpy(labels).reshape(-1).astype(bool)
    prediction = _numpy(scores).reshape(-1).astype(float)
    positive = prediction[truth]
    negative = prediction[~truth]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def binary_average_precision(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute binary average precision.

    Args:
        labels: Binary ground-truth labels.
        scores: Continuous prediction scores.

    Returns:
        Average precision, or NaN when no positive label is present.
    """
    truth = _numpy(labels).reshape(-1).astype(bool)
    prediction = _numpy(scores).reshape(-1).astype(float)
    if not truth.any():
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(truth, prediction))


def compute_multilabel_metrics(labels: ArrayLike, scores: ArrayLike) -> dict[str, object]:
    """Compute per-label and macro AUROC/AP.

    Args:
        labels: Binary matrix shaped ``(N,L)``.
        scores: Prediction matrix with the same shape.

    Returns:
        Per-label arrays and finite-label macro averages.
    """
    truth = _numpy(labels)
    prediction = _numpy(scores)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("labels and scores must be matching matrices shaped (N,L).")
    auroc = np.asarray([binary_auroc(truth[:, i], prediction[:, i]) for i in range(truth.shape[1])])
    average_precision = np.asarray(
        [binary_average_precision(truth[:, i], prediction[:, i]) for i in range(truth.shape[1])]
    )
    return {
        "auroc_per_label": auroc.tolist(),
        "ap_per_label": average_precision.tolist(),
        "mean_auroc": float(np.nanmean(auroc)) if np.isfinite(auroc).any() else float("nan"),
        "mean_ap": float(np.nanmean(average_precision)) if np.isfinite(average_precision).any() else float("nan"),
    }


def classify_volumes(paths, case_ids, labels_csv, checkpoint, *, slice_axis=0, spatial_pool_size=14):
    """Restore CT-Net, normalize HU, group 126 slices into 42 RGB triplets.

    Labels CSV: VolumeName (or case_id), followed by the checkpoint's label names.
    Missing labels are errors; no random-initialized scores are ever reported.
    """
    import csv
    import nibabel as nib
    from gexco.evaluation.ctnet import CTNetModel
    from gexco.data.io import case_id

    if not checkpoint:
        raise ValueError("A trained CT-Net checkpoint is required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = next((payload[k] for k in ("model", "params", "state_dict", "model_state_dict") if k in payload), None)
    if state is None:
        raise KeyError("CT-Net checkpoint has no model state")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        id_column = "VolumeName" if "VolumeName" in reader.fieldnames else "case_id"
        columns = [c for c in reader.fieldnames if c != id_column]
        rows = {case_id(row[id_column]): row for row in reader}
    trained_labels = payload.get("label_columns", columns)
    if any(c not in trained_labels for c in columns):
        raise ValueError("CSV contains labels outside the classifier ontology")
    label_names = [c for c in trained_labels if c in columns]
    if not label_names:
        raise ValueError("No common classifier labels")
    score_indices = [trained_labels.index(c) for c in label_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTNetModel(len(trained_labels), spatial_pool_size=spatial_pool_size)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    truth = []
    scores = []
    with torch.inference_mode():
        for path, name in zip(paths, case_ids, strict=True):
            if name not in rows:
                raise ValueError(f"Missing classification labels: {name}")
            volume = nib.load(str(path)).get_fdata(dtype=np.float32)
            if volume.shape != (128, 128, 128):
                raise ValueError("CT-Net requires CT128")
            volume = np.moveaxis(volume, int(slice_axis), 0)
            volume = (np.clip(volume, -1000.0, 1000.0) + 1000.0) / 2000.0
            volume = volume[:126].reshape(42, 3, 128, 128)
            scores.append(
                torch.sigmoid(model(torch.from_numpy(volume.copy())[None].to(device)))[0].cpu().numpy()[score_indices]
            )
            truth.append([float(rows[name][label]) for label in label_names])
    metrics = compute_multilabel_metrics(np.asarray(truth), np.asarray(scores))
    return {
        **metrics,
        "label_columns": label_names,
        "case_ids": case_ids,
        "scores": np.asarray(scores).tolist(),
        "num_samples": len(paths),
        "used_random_init": False,
    }
