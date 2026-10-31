"""Download model checkpoints with optional SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def download(url: str, destination: Path, sha256: str | None = None) -> Path:
    """Download one checkpoint atomically.

    Args:
        url: Public checkpoint URL.
        destination: Final local path.
        sha256: Optional expected hexadecimal digest.

    Returns:
        Downloaded checkpoint path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if sha256:
        hasher = hashlib.sha256()
        with temporary.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest.lower() != sha256.lower():
            temporary.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {destination.name}: {digest}")
    temporary.replace(destination)
    return destination


def main() -> None:
    """Parse checkpoint URLs and download any that were supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--gexco", action="store_true", help="Download the released GeXCo and Any-Anchor checkpoints")
    parser.add_argument(
        "--maisi", action="store_true", help="Download the official epoch-273 MAISI VAE under NVIDIA terms"
    )
    parser.add_argument("--world-model-url")
    parser.add_argument("--world-model-sha256")
    parser.add_argument("--reconstruction-url")
    parser.add_argument("--reconstruction-sha256")
    arguments = parser.parse_args()
    if arguments.gexco and (arguments.world_model_url or arguments.reconstruction_url):
        parser.error("Use --gexco or custom checkpoint URLs, not both.")
    requested = []
    if arguments.gexco:
        manifest = json.loads((Path(__file__).resolve().parents[1] / "gexco/weights/release.json").read_text())
        requested.extend(
            (info["url"], arguments.output_dir / name, info["sha256"])
            for name, info in manifest["assets"].items()
        )
    if arguments.maisi:
        requested.append(
            (
                "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
                arguments.output_dir / "autoencoder_epoch273.pt",
                "1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3",
            )
        )
    requested += [
        (
            arguments.world_model_url,
            arguments.output_dir / "world_model_best.pt",
            arguments.world_model_sha256,
        ),
        (
            arguments.reconstruction_url,
            arguments.output_dir / "gexco.pt",
            arguments.reconstruction_sha256,
        ),
    ]
    downloads = [download(url, path, digest) for url, path, digest in requested if url]
    if not downloads:
        parser.error("Use --gexco, --maisi or provide --world-model-url and/or --reconstruction-url.")
    for path in downloads:
        print(path)


if __name__ == "__main__":
    main()
