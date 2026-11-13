"""Pack per-case MAISI latent NIfTIs into indexed HDF5 shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Optional

import h5py
import nibabel as nib
import numpy as np


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text())
        records = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Input manifest must contain a list of sample records.")
    return records


def _latent_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*.nii.gz"):
        case_id = path.name.removesuffix(".nii.gz")
        if case_id in index:
            raise ValueError(f"Duplicate latent case ID {case_id}: {index[case_id]} and {path}")
        index[case_id] = path
    return index


def _read_latent(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.shape == (32, 32, 32, 4):
        array = np.moveaxis(array, -1, 0)
    if array.shape != (4, 32, 32, 32) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite (4,32,32,32) latent in {path}, got {array.shape}.")
    return array


def _bank_records(bank_root: Path, split: str, ct_root: Path, drr_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    shard_dir = bank_root / split
    for shard in sorted(shard_dir.glob("*.h5")):
        with h5py.File(shard, "r") as h5:
            angles = [float(angle) for angle in h5.attrs["angles_deg"]]
            angle_indices = [angles.index(0.0), angles.index(90.0)]
            case_key = "index/case_id" if "index/case_id" in h5 else "index/patient_id"
            case_ids = h5[case_key].asstr()[:]
            patient_ids = h5["index/patient_id"].asstr()[:] if "index/patient_id" in h5 else None
            for row, case_id in enumerate(case_ids):
                if case_id in seen:
                    raise ValueError(f"Duplicate case ID in DRR bank: {case_id}")
                seen.add(case_id)
                parts = str(case_id).split("_")
                if len(parts) < 4 or parts[0] != split:
                    raise ValueError(f"Unexpected CT-RATE case ID for {split}: {case_id}")
                patient_dir = "_".join(parts[:2])
                series_dir = "_".join(parts[:3])
                relative_ct = Path(split) / patient_dir / series_dir / f"{case_id}.nii.gz"
                bank_shard = drr_root / split / shard.name
                records.append(
                    {
                        "case_id": str(case_id),
                        "patient_id": str(patient_ids[row]) if patient_ids is not None else patient_dir,
                        "volume": str(ct_root / relative_ct),
                        "views": [
                            {
                                "path": f"h5://{bank_shard}::{row}::{view_index}::128",
                                "angle_deg": angles[view_index],
                            }
                            for view_index in angle_indices
                        ],
                        "spacing_mm": [3.0, 3.0, 3.0],
                    }
                )
    if not records:
        raise ValueError(f"No HDF5 shards found under {shard_dir}")
    return records


def pack(
    manifest: Optional[Path],
    latent_dir: Path,
    output_dir: Path,
    *,
    shard_size: int = 512,
    bank_root: Optional[Path] = None,
    split: Optional[str] = None,
    ct_root: Optional[Path] = None,
    drr_root: Optional[Path] = None,
    workers: int = 4,
) -> Path:
    """Pack latents referenced by a training manifest and write an updated manifest."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if manifest is not None:
        records = _read_manifest(manifest)
        manifest_dir = manifest.parent
    else:
        if bank_root is None or split not in {"train", "valid"} or ct_root is None or drr_root is None:
            raise ValueError("Supply --manifest or all of --drr-bank-root, --split, --ct-root, and --drr-path-root")
        records = _bank_records(bank_root, split, ct_root, drr_root)
        manifest_dir = Path.cwd()
    if not records:
        raise ValueError("Input manifest has no samples")

    resolved: list[tuple[dict[str, Any], Path]] = []
    seen: set[str] = set()
    latent_files: Optional[dict[str, Path]] = None
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case_id in manifest: {case_id!r}")
        seen.add(case_id)
        latent_value = record.get("latent")
        if isinstance(latent_value, dict):
            candidate = latent_value.get("path", latent_value.get("hdf5"))
        else:
            candidate = latent_value
        if candidate:
            source = Path(candidate)
            if not source.is_absolute():
                source = manifest_dir / source
        else:
            parts = case_id.split("_")
            if len(parts) >= 4:
                source = latent_dir / parts[0] / "_".join(parts[:2]) / "_".join(parts[:3]) / f"{case_id}.nii.gz"
            else:
                if latent_files is None:
                    latent_files = _latent_index(latent_dir)
                source = latent_files.get(case_id, Path())
        resolved.append((record, source))

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    updated: list[dict[str, Any]] = []
    shard_count = (len(resolved) + shard_size - 1) // shard_size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shard_number, start in enumerate(range(0, len(resolved), shard_size)):
            subset = resolved[start : start + shard_size]
            shard_name = f"latents_{shard_number:05d}.h5"
            shard_path = shard_dir / shard_name
            partial = shard_path.with_suffix(".h5.partial")
            if shard_path.exists():
                with h5py.File(shard_path, "r") as h5:
                    existing_ids = h5["index/case_id"].asstr()[:].tolist()
                expected_ids = [str(record["case_id"]) for record, _ in subset]
                if existing_ids != expected_ids:
                    raise ValueError(f"Existing shard does not match manifest order: {shard_path}")
            else:
                partial.unlink(missing_ok=True)
                try:
                    values = np.stack(list(pool.map(_read_latent, (source for _, source in subset))))
                except FileNotFoundError as error:
                    raise FileNotFoundError(f"Latent file missing while packing shard {shard_name}: {error}") from error
                with h5py.File(partial, "w") as h5:
                    h5.attrs.update(
                        format="gexco_maisi_latents_v1", latent_shape=[4, 32, 32, 32],
                        source_format="NIfTI float32"
                    )
                    h5.create_dataset(
                        "latents", data=values, dtype="float32", chunks=(1, 4, 32, 32, 32),
                        compression="gzip", compression_opts=1, shuffle=True
                    )
                    h5.create_dataset(
                        "index/case_id", data=[r["case_id"] for r, _ in subset], dtype=h5py.string_dtype()
                    )
                partial.replace(shard_path)
                print(f"Packed {shard_number + 1}/{shard_count} shards: {shard_name}", flush=True)
            relative_shard = str(Path("shards") / shard_name)
            for row, (record, _) in enumerate(subset):
                updated.append({**record, "latent": {"hdf5": relative_shard, "row": row}})

    out_manifest = output_dir / "manifest.json"
    out_manifest.write_text(json.dumps({"samples": updated}, indent=2) + "\n")
    return out_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Training manifest with case IDs and DRR references")
    parser.add_argument("--latent-dir", type=Path, required=True, help="Root containing per-case MAISI .nii.gz files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4, help="Parallel latent readers per shard")
    parser.add_argument("--drr-bank-root", type=Path, help="Source HDF5 bank with train/ and valid/ subfolders")
    parser.add_argument("--split", choices=("train", "valid"))
    parser.add_argument("--ct-root", type=Path, help="CT128 destination root to record in output manifests")
    parser.add_argument("--drr-path-root", type=Path, help="DRR HDF5 destination root to record in output manifests")
    args = parser.parse_args()
    print(pack(args.manifest, args.latent_dir, args.output_dir, shard_size=args.shard_size,
               bank_root=args.drr_bank_root, split=args.split, ct_root=args.ct_root, drr_root=args.drr_path_root,
               workers=args.workers))


if __name__ == "__main__":
    main()
