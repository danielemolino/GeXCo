"""Paper preprocessing of acquired biplanar DICOM radiographs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut
from scipy import ndimage
import tifffile as tiff
import torch
import torch.nn.functional as F

from gexco.config import default_config, load_config


def norm_for_mask(array: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(array, [1, 99])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0, 1).astype(np.float32)


def foreground_mask(array: np.ndarray) -> np.ndarray:
    """Largest thresholded connected body component after polarity correction."""
    n = norm_for_mask(array)
    mask = n > 0.05
    # Close small detector/background gaps, then retain the largest component.
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.ones_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    return ndimage.binary_fill_holes(mask)


def resample_to_iso(array: np.ndarray, spacing_row: float, spacing_col: float, target_spacing_mm: float) -> tuple[np.ndarray, float, float]:
    rows, cols = array.shape
    out_rows = max(1, int(round(rows * spacing_row / target_spacing_mm)))
    out_cols = max(1, int(round(cols * spacing_col / target_spacing_mm)))
    x = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    y = F.interpolate(x, size=(out_rows, out_cols), mode="area")[0, 0].numpy()
    effective_row = rows * spacing_row / out_rows
    effective_col = cols * spacing_col / out_cols
    return y.astype(np.float32), effective_row, effective_col


def center_bbox_to_square(array: np.ndarray, mask: np.ndarray, side: int) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        bbox = (0, array.shape[0], 0, array.shape[1])
    else:
        bbox = (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)
    cy = (bbox[0] + bbox[1] - 1) / 2.0
    cx = (bbox[2] + bbox[3] - 1) / 2.0
    offset_y = int(round((side - 1) / 2.0 - cy))
    offset_x = int(round((side - 1) / 2.0 - cx))
    out = np.full((side, side), float(np.min(array)), dtype=np.float32)
    src_y0, src_x0 = max(0, -offset_y), max(0, -offset_x)
    dst_y0, dst_x0 = max(0, offset_y), max(0, offset_x)
    h = min(array.shape[0] - src_y0, side - dst_y0)
    w = min(array.shape[1] - src_x0, side - dst_x0)
    if h > 0 and w > 0:
        out[dst_y0:dst_y0+h, dst_x0:dst_x0+w] = array[src_y0:src_y0+h, src_x0:src_x0+w]
    return out, bbox, offset_y, offset_x


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def resolve_geometry(ds, view: str, config: dict) -> dict:
    """Resolve detector pitch and approximate magnification from DX metadata."""
    if view not in {"frontal", "lateral"}:
        raise ValueError("Specify frontal or lateral explicitly; ViewPosition is not inferred")
    group = "|".join(str(getattr(ds, key, "")) for key in
                     ("Manufacturer", "ManufacturerModelName", "Rows", "Columns"))
    tagged = getattr(ds, "ImagerPixelSpacing", None)
    if tagged is not None and len(tagged) == 2:
        spacing = [_positive(v, "ImagerPixelSpacing") for v in tagged]
        tier = "a_imager_pixel_spacing"
    elif group in config["detector_spacing_donors_mm"]:
        spacing = [_positive(config["detector_spacing_donors_mm"][group], "Donor spacing")] * 2
        tier = "b_manufacturer_model_rows_columns"
    else:
        spacing = [_positive(config["detector_spacing_default_mm"], "Default spacing")] * 2
        tier = "c_global_median_default"
    sid = getattr(ds, "DistanceSourceToDetector", None)
    sid_imputed = sid is None or str(sid) == ""
    sid = _positive(config["source_detector_default_mm"] if sid_imputed else sid, "Source-detector distance")
    support = getattr(ds, "DistanceSourceToPatient", None)
    support_imputed = support is None or str(support) == ""
    support = _positive(sid - float(config["source_support_gap_default_mm"]) if support_imputed else support,
                        "Source-support distance")
    midpoint = _positive(support - float(config["thorax_depth_mm"][view]) / 2, "Source-midplane distance")
    magnification = sid / midpoint
    return {
        "pixel_spacing_tier": tier,
        "pixel_spacing_tier_c": tier.startswith("c_"),
        "detector_spacing_row_col_mm": spacing,
        "source_detector_mm": sid,
        "source_support_mm": support,
        "source_detector_imputed": sid_imputed,
        "source_support_imputed": support_imputed,
        "magnification_factor": magnification,
        "isocenter_spacing_row_col_mm": [v / magnification for v in spacing],
    }


def preprocess_dicom(path: str | Path, view: str, config: dict | None = None):
    """Return normalized 512/128 arrays and preprocessing decisions, using only the radiograph."""
    config = config if config is not None else load_config(default_config("radiographs"))
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise ValueError("Acquired inputs require DICOM metadata; use normalized for prepared [0,1] TIFFs")
    ds = pydicom.dcmread(path)
    photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("A single-frame monochrome DICOM radiograph is required")
    modality = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
    if modality.ndim != 2 or not np.isfinite(modality).any():
        raise ValueError("A finite, two-dimensional radiograph is required")
    if photo == "MONOCHROME1":
        modality = float(np.nanmin(modality) + np.nanmax(modality)) - modality
    modality = np.nan_to_num(modality, nan=float(np.nanmin(modality)),
                            posinf=float(np.nanmax(modality)), neginf=float(np.nanmin(modality)))
    if not np.isfinite(modality).all():
        raise ValueError("Radiograph contains invalid intensity bounds")
    decisions = resolve_geometry(ds, view, config)
    spacing = decisions["isocenter_spacing_row_col_mm"]
    target = _positive(config["target_isocenter_spacing_mm"], "Target spacing")
    scaled, eff_row, eff_col = resample_to_iso(modality, *spacing, target)
    mask = foreground_mask(scaled)
    side = int(config["output_size"])
    if side != 512:
        raise ValueError("The paper preprocessing requires output_size: 512")
    centered, bbox, offset_y, offset_x = center_bbox_to_square(scaled, mask, side)
    src_p1, src_p99 = (float(v) for v in np.percentile(scaled[mask], [1, 99]))
    if src_p99 <= src_p1:
        raise ValueError("Degenerate radiograph body intensity range")
    stats = json.loads(Path(config["stats_path"]).read_text())["train_global"]
    train_p1, train_p99 = float(stats["p1"]), float(stats["p99"])
    if not np.isfinite([train_p1, train_p99]).all() or train_p99 <= train_p1:
        raise ValueError("Invalid fixed training DRR percentiles")
    mapped = (centered - src_p1) * ((train_p99 - train_p1) / (src_p99 - src_p1)) + train_p1
    mapped = np.clip(mapped, train_p1, train_p99).astype(np.float32)
    normalized = ((np.clip(mapped, train_p1, train_p99) - train_p1) / (train_p99 - train_p1)).astype(np.float32)
    down128 = F.interpolate(torch.from_numpy(normalized)[None, None], size=(128, 128), mode="area")[0, 0].numpy()
    decisions.update({
        "view": view,
        "monochrome1_inverted": photo == "MONOCHROME1",
        "target_isocenter_spacing_mm": target,
        "effective_resampled_spacing_row_col_mm": [eff_row, eff_col],
        "resampled_shape": list(scaled.shape),
        "body_bbox_y0_y1_x0_x1": list(bbox),
        "center_offset_y_x": [offset_y, offset_x],
        "body_p1_raw": src_p1, "body_p99_raw": src_p99,
        "training_p1_raw": train_p1, "training_p99_raw": train_p99,
    })
    return {512: normalized, 128: down128.astype(np.float32)}, decisions


def load_acquired(path, image_size, *, view, config=None):
    arrays, _ = preprocess_dicom(path, view, config)
    if image_size not in arrays:
        raise ValueError("Paper radiograph inputs are 128 or 512 pixels")
    return torch.from_numpy(arrays[image_size]).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config("radiographs"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSON sample list with case_id, frontal and lateral DICOM paths")
    source.add_argument("--frontal", type=Path)
    parser.add_argument("--lateral", type=Path)
    parser.add_argument("--case-id", default="case_0001")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frontal and not args.lateral:
        parser.error("--lateral is required with --frontal")
    config = load_config(args.config)
    if args.manifest:
        payload = json.loads(args.manifest.read_text())
        records = payload.get("samples", payload) if isinstance(payload, dict) else payload
    else:
        records = [{"case_id": args.case_id, "frontal": str(args.frontal), "lateral": str(args.lateral)}]
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty list of frontal/lateral pairs")
    case_ids = [str(r["case_id"]) for r in records]
    if len(set(case_ids)) != len(case_ids) or any(Path(c).name != c or c in {"", ".", ".."} for c in case_ids):
        raise ValueError("case_id must be unique and a plain filename")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    samples, decisions = [], []
    for record, case_id in zip(records, case_ids):
        views = []
        case_decisions = []
        for view, suffix, angle in (("frontal", "F", 0), ("lateral", "L", 90)):
            source_path = Path(record[view])
            arrays, decision = preprocess_dicom(source_path, view, config)
            decision.update(case_id=case_id, source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
            for size, array in arrays.items():
                target = out / str(size) / case_id / f"{case_id}_{suffix}.tiff"
                target.parent.mkdir(parents=True, exist_ok=True)
                tiff.imwrite(target, array)
            views.append({"path": str(out / "128" / case_id / f"{case_id}_{suffix}.tiff"), "angle_deg": angle})
            case_decisions.append(decision)
        tier_c = any(d["pixel_spacing_tier_c"] for d in case_decisions)
        for d in case_decisions:
            d["patient_has_any_tier_c_view"] = tier_c
        decisions.extend(case_decisions)
        samples.append({"case_id": case_id, "views": views})
    (out / "manifest.json").write_text(json.dumps({"samples": samples}, indent=2) + "\n")
    (out / "preprocessing.json").write_text(json.dumps({"config": config, "views": decisions}, indent=2) + "\n")
    print(f"Prepared {len(samples)} pairs; infer --input-kind normalized --manifest {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
