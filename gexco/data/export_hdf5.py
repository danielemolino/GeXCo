"""Export DiffDRR projections to atomic, indexed multiview HDF5 shards."""

import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
from gexco.config import default_config, load_config, normalize_density_mode
from gexco.data.drr_rendering import render_ct
from gexco.data.hdf5 import manifest_from_hdf5
from gexco.pretraining.data import read_manifest


def export(records, output_dir, config):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    angles = config["drr"]["angles_deg"]
    size = int(config["drr"]["output_size"])
    pitch = float(config["drr"]["pixel_size_mm"]) * int(config["drr"]["detector_size"]) / size
    shard_size = int(config["hdf5"]["shard_size"])
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    ids = [str(row["case_id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    for offset in range(0, len(records), shard_size):
        subset = records[offset : offset + shard_size]
        final = root / f"shard_{offset // shard_size:05d}.h5"
        if final.exists():
            raise FileExistsError(final)
        partial = final.with_suffix(".h5.partial")
        with h5py.File(partial, "w") as h5:
            h5.attrs.update(
                density_mode=normalize_density_mode(config["drr"]["density_mode"]),
                renderer="diffdrr",
                renderer_version="0.6.0",
                rotate_canonical_k=0,
                angles_deg=np.asarray(angles, dtype="float32"),
                sdd=config["drr"]["source_to_detector_mm"],
                sdp=config["drr"]["source_to_patient_mm"],
                delx_128=float(config["drr"]["pixel_size_mm"]) * int(config["drr"]["detector_size"]) / 128,
                effective_pixel_size_mm=pitch,
                output_size=size,
                index_schema_version=2,
                stored_sampling=config["drr"]["stored_sampling"],
            )
            index = h5.create_group("index")
            index.create_dataset("case_id", data=[str(x["case_id"]) for x in subset], dtype=h5py.string_dtype())
            index.create_dataset(
                "patient_id",
                data=[str(x["patient_id"]) if x.get("patient_id") is not None else "" for x in subset],
                dtype=h5py.string_dtype(),
            )
            index.create_dataset(
                "ct_path", data=[str(Path(x["volume"]).resolve()) for x in subset], dtype=h5py.string_dtype()
            )
            complete = index.create_dataset("complete", data=np.zeros(len(subset), dtype=bool))
            images = h5.create_dataset(
                f"views/{size}",
                shape=(len(subset), len(angles), size, size),
                dtype="float32",
                chunks=(1, 1, size, size),
            )
            for row, record in enumerate(subset):
                images[row] = render_ct(record["volume"], angles, config["drr"])
                complete[row] = True
        os.replace(partial, final)
    manifests = {}
    for biplanar, filename in [(False, "world_model_manifest.json"), (True, "biplanar_manifest.json")]:
        exported = manifest_from_hdf5(root, biplanar=biplanar, resolution=size)
        by_id = {x["case_id"]: x for x in records}
        for record in exported:
            source = by_id[record["case_id"]]
            record.update({key: value for key, value in source.items() if key != "views"})
        path = root / filename
        path.write_text(json.dumps({"samples": exported}, indent=2) + "\n")
        manifests[filename] = str(path)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL: case_id and preprocessed volume path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=default_config("preparation"))
    args = parser.parse_args()
    print(json.dumps(export(read_manifest(args.manifest), args.output_dir, load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
