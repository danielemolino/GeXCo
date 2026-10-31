"""TotalSegmentator 2.18.0 evaluation of 29 non-rib labels."""

import importlib.metadata
from pathlib import Path
import numpy as np
import torch

THORAX_CLASS_IDS = {
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    79: "spinal_cord",
    116: "sternum",
}
assert len(THORAX_CLASS_IDS) == 29
MAX_HD95_MM = float(384.0 * np.sqrt(3.0))  # Voxel-support diagonal, not 381-mm centre diagonal.


def _boolean_array(value):
    return (
        value.detach().cpu().numpy().astype(bool) if isinstance(value, torch.Tensor) else np.asarray(value, dtype=bool)
    )


def dice_score(prediction, target, *, empty_value=float("nan")):
    p, t = _boolean_array(prediction), _boolean_array(target)
    if p.shape != t.shape:
        raise ValueError("Mask grids differ")
    n = int(p.sum() + t.sum())
    return float(2 * np.logical_and(p, t).sum() / n) if n else float(empty_value)


def hd95(prediction, target, *, spacing=(3.0, 3.0, 3.0)):
    p, t = _boolean_array(prediction), _boolean_array(target)
    if p.shape != t.shape:
        raise ValueError("Mask grids differ")
    if not p.any() and not t.any():
        return float("nan")
    if not p.any() or not t.any():
        return MAX_HD95_MM
    from monai.metrics import compute_hausdorff_distance

    return float(
        compute_hausdorff_distance(
            torch.from_numpy(p[None, None]),
            torch.from_numpy(t[None, None]),
            include_background=True,
            percentile=95.0,
            directed=False,
            spacing=spacing,
        ).item()
    )


def segmentation_metrics(prediction, target, *, spacing=(3.0, 3.0, 3.0)):
    p, t = _boolean_array(prediction), _boolean_array(target)
    if p.shape != t.shape:
        raise ValueError("Mask grids differ")
    if not p.any() and not t.any():
        return {"dice": None, "hd95": None, "nsd_3mm": None}
    if not p.any() or not t.any():
        return {"dice": 0.0, "hd95": MAX_HD95_MM, "nsd_3mm": 0.0}
    from monai.metrics.surface_dice import compute_surface_dice

    nsd = compute_surface_dice(
        torch.from_numpy(p[None, None]),
        torch.from_numpy(t[None, None]),
        class_thresholds=[3.0],
        include_background=True,
        distance_metric="euclidean",
        spacing=spacing,
        use_subvoxels=True,
    )
    return {"dice": dice_score(p, t), "hd95": hd95(p, t, spacing=spacing), "nsd_3mm": float(nsd.item())}


def label_metrics(prediction, target, *, spacing=(3.0, 3.0, 3.0)):
    per_structure = {}
    for class_id, name in THORAX_CLASS_IDS.items():
        p, t = prediction == class_id, target == class_id
        values = segmentation_metrics(p, t, spacing=spacing)
        per_structure[name] = values
    means = {
        key: float(np.mean([row[key] for row in per_structure.values() if row[key] is not None]))
        if any(row[key] is not None for row in per_structure.values())
        else None
        for key in ["dice", "hd95", "nsd_3mm"]
    }
    return {"summary": means, "per_structure": per_structure, "missing_structure_hd95_mm": MAX_HD95_MM}


def segment_with_totalsegmentator(input_volume, output_path, *, device="gpu", task="total"):
    if importlib.metadata.version("TotalSegmentator") != "2.18.0":
        raise RuntimeError("Paper segmentation requires TotalSegmentator==2.18.0")
    from totalsegmentator.python_api import totalsegmentator
    from gexco.utils.tensors import seed_everything

    seed_everything(0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        str(input_volume),
        str(output_path),
        task=task,
        ml=True,
        fast=False,
        fastest=False,
        device=device,
        roi_subset=list(THORAX_CLASS_IDS.values()),
        nr_thr_resamp=1,
        nr_thr_saving=1,
        quiet=True,
    )


dice = dice_score
