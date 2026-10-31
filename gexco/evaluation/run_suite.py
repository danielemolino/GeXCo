"""Method-agnostic evaluation of paired HU NIfTI folders (or a case manifest)."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from gexco.config import default_config, load_config
from gexco.data.io import case_id
from gexco.utils.json import json_dumps
from gexco.evaluation.volume_metrics import volume_metrics
from gexco.evaluation.segmentation import label_metrics, segmentation_metrics, segment_with_totalsegmentator


def _limit_paths(paths, max_cases):
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive")
    return paths if max_cases is None else paths[:max_cases]


def _index(root):
    index = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and (path.name.endswith((".nii", ".nii.gz")) or path.suffix == ".npy"):
            name = case_id(path)
            if name in index:
                raise ValueError(f"Ambiguous case ID {name}; use --manifest")
            index[name] = path
    return index


def load_pair(prediction, reference):
    if Path(prediction).suffix == ".npy" and Path(reference).suffix == ".npy":
        return np.load(prediction).squeeze(), np.load(reference).squeeze(), (3.0, 3.0, 3.0)
    p, t = nib.load(str(prediction)), nib.load(str(reference))
    if p.shape != t.shape or not np.allclose(p.affine, t.affine, atol=1e-5):
        raise ValueError(f"Prediction/reference grids differ: {prediction}; resample explicitly before scoring")
    return p.get_fdata(dtype=np.float32), t.get_fdata(dtype=np.float32), tuple(map(float, t.header.get_zooms()[:3]))


def _summarize(cases):
    keys = [k for k in cases[0] if k != "case_id"]
    return {
        key: float(np.mean([row[key] for row in cases if row[key] is not None]))
        if any(row[key] is not None for row in cases)
        else None
        for key in keys
    }


def paired_records(predictions, targets, max_cases=None):
    p, t = _index(predictions), _index(targets)
    names = _limit_paths(sorted(p), max_cases)
    if not names:
        raise ValueError("No predictions found")
    if any(name not in t for name in names):
        raise ValueError("Missing references for predictions")
    return [{"case_id": name, "prediction": str(p[name]), "reference": str(t[name])} for name in names]


def evaluate_volume_directories(prediction_dir, target_dir, *, max_cases=None):
    cases = []
    for record in paired_records(prediction_dir, target_dir, max_cases):
        p, t, _ = load_pair(record["prediction"], record["reference"])
        cases.append({"case_id": record["case_id"], **volume_metrics(p, t)})
    return {"summary": {"num_cases": len(cases), **_summarize(cases)}, "cases": cases}


def evaluate_mask_directories(prediction_dir, target_dir, *, max_cases=None):
    cases = []
    for record in paired_records(prediction_dir, target_dir, max_cases):
        p, t, spacing = load_pair(record["prediction"], record["reference"])
        cases.append({"case_id": record["case_id"], **segmentation_metrics(p, t, spacing=spacing)})
    return {"summary": {"num_cases": len(cases), **_summarize(cases)}, "cases": cases}


def _mask_for(path, cache, device):
    path = Path(path)
    identity = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}:TotalSegmentator2.18.0:29:seed0"
    key = hashlib.sha256(identity.encode()).hexdigest()
    destination = cache / f"{key}.nii.gz"
    if not destination.exists():
        segment_with_totalsegmentator(path, destination, device=device)
    mask = nib.load(str(destination))
    image = nib.load(str(path))
    if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, atol=1e-5):
        raise ValueError("Segmentation cache grid mismatch")
    return np.asarray(mask.dataobj)


def evaluate(records, config, output_dir, groups):
    from gexco.evaluation.drr_metrics import reprojection_metrics
    from gexco.data.drr_rendering import render_ct
    from gexco.evaluation.frd import frd_for_volumes
    from gexco.evaluation.classification import classify_volumes

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    segmentation = []
    prep = load_config(default_config("preparation"))
    reconstruction = load_config(default_config("reconstruction"))
    for record in records:
        p, t, spacing = load_pair(record["prediction"], record["reference"])
        row = {"case_id": record["case_id"]}
        if "volume" in groups:
            row["volume"] = volume_metrics(p, t)
        if "reprojection" in groups:
            # render_ct performs the same HU clipping and DiffDRR density conversion
            # on prediction and reference. Normalize both with fixed TRAIN percentiles.
            predicted = render_ct(record["prediction"], [0.0, 90.0], prep["drr"])
            target = render_ct(record["reference"], [0.0, 90.0], prep["drr"])
            stats = json.loads(Path(reconstruction["drr"]["stats_path"]).read_text())["train_global"]
            lo, hi = float(stats["p1"]), float(stats["p99"])
            predicted = np.clip((predicted - lo) / (hi - lo), 0, 1)
            target = np.clip((target - lo) / (hi - lo), 0, 1)
            row["reprojection"] = reprojection_metrics(list(predicted), list(target), data_range=1.0)
        if "segmentation" in groups:
            if p.shape != (128, 128, 128) or not np.allclose(spacing, (3, 3, 3)):
                raise ValueError("Segmentation evaluation requires CT128 at 3 mm")
            pred = _mask_for(record["prediction"], root / "segmentation_cache", config["segmentation"]["device"])
            ref = _mask_for(record["reference"], root / "segmentation_cache", config["segmentation"]["device"])
            metrics = label_metrics(pred, ref, spacing=spacing)
            row["segmentation"] = metrics["summary"]
            segmentation.append({"case_id": record["case_id"], **metrics})
        cases.append(row)
        print(json.dumps({"case": record["case_id"], "completed_metrics": list(row)[1:]}), flush=True)
    result = {"num_cases": len(records), "cases": cases, "summary": {}}
    for group in ("volume", "reprojection", "segmentation"):
        if group in groups:
            result["summary"][group] = _summarize([{"case_id": row["case_id"], **row[group]} for row in cases])
    if segmentation:
        result["segmentation"] = segmentation
    if "frd" in groups:
        result["summary"]["frd"] = frd_for_volumes(
            [r["reference"] for r in records],
            [r["prediction"] for r in records],
            root / "frd_cache",
            num_workers=config["frd"]["num_workers"],
        )
    if "classification" in groups:
        c = config["classification"]
        result["classification"] = classify_volumes(
            [r["prediction"] for r in records],
            [r["case_id"] for r in records],
            c["labels"],
            c["checkpoint"],
            slice_axis=c["slice_axis"],
            spatial_pool_size=c["spatial_pool_size"],
        )
    if "classification" in result:
        c = result["classification"]
        result["summary"]["classification"] = {key: c[key] for key in ("mean_auroc", "mean_ap", "num_samples")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config("evaluation"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument(
        "--manifest", type=Path, help="List of case_id, prediction, reference paths; supports any folder layout"
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=["volume", "reprojection", "segmentation", "frd", "classification"]
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--classification-checkpoint", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.manifest:
        from gexco.pretraining.data import read_manifest

        records = read_manifest(args.manifest)
        records = _limit_paths(sorted(records, key=lambda x: x["case_id"]), args.max_cases)
    else:
        if not args.predictions or not args.targets:
            parser.error("Supply --predictions and --targets or --manifest")
        records = paired_records(args.predictions, args.targets, args.max_cases)
    if not records or len({x["case_id"] for x in records}) != len(records):
        raise ValueError("Empty or duplicate case IDs")
    if args.classification_checkpoint:
        config["classification"]["checkpoint"] = str(args.classification_checkpoint)
    if args.labels:
        config["classification"]["labels"] = str(args.labels)
    result = evaluate(records, config, args.output_dir, args.metrics or config["metrics"])
    (args.output_dir / "metrics.json").write_text(json_dumps(result, indent=2) + "\n")
    print(json_dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
