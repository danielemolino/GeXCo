"""Extract unscaled, frozen MAISI VAE latents and an updated training manifest."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gexco.config import default_config, load_config
from gexco.data.io import load_ct
from gexco.pretraining.data import read_manifest
from gexco.reconstruction.autoencoder import load_maisi_autoencoder
from gexco.utils.tensors import seed_everything


@torch.inference_mode()
def extract(records, output_dir, checkpoint, config):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_maisi_autoencoder(checkpoint, device=device)
    dtype = next(autoencoder.parameters()).dtype
    updated = []
    for record in records:
        # Reset per case: sharding and manifest order do not change posterior noise.
        seed_everything(int(config["latents"]["seed"]))
        array, affine = load_ct(record["volume"])
        tensor = torch.from_numpy((array + 1000.0) / 2000.0)[None, None].to(device=device, dtype=dtype)
        if config["latents"]["sample_posterior"]:
            latent = autoencoder.encode_stage_2_inputs(tensor)
        else:
            latent, _ = autoencoder.encode(tensor)
        latent = latent[0].float().cpu().numpy()
        if latent.shape != (4, 32, 32, 32) or not np.isfinite(latent).all():
            raise ValueError(f"Unexpected MAISI latent: {latent.shape}")
        path = root / f"{record['case_id']}.npy"
        np.save(path, latent)
        updated.append(
            {**record, "latent": str(path.resolve()), "spacing_mm": [3.0, 3.0, 3.0], "affine": affine.tolist()}
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"samples": updated}, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=default_config("preparation"))
    args = parser.parse_args()
    print(extract(read_manifest(args.manifest), args.output_dir, args.vae_checkpoint, load_config(args.config)))


if __name__ == "__main__":
    main()
