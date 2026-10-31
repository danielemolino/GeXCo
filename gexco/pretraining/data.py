"""Manifest-driven multi-angle DRR dataset for GeXCo world-model pretraining."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_grayscale_image(
    path: str | Path,
    image_size: int = 128,
    *,
    stats_path: str | Path | None = None,
    normalization: str = "per_image_minmax",
) -> torch.Tensor:
    """Load and robustly normalize a grayscale DRR.

    Args:
        path: Image, NumPy, or PyTorch tensor path.
        image_size: Square output resolution.
        stats_path: JSON containing ``train_global.p1`` and ``p99``.
        normalization: ``dataset_percentile_fixed`` for checkpoint-exact
            normalization, ``per_image_minmax`` for standalone examples, or
            ``none`` for already normalized inputs.

    Returns:
        Float tensor shaped ``(1,image_size,image_size)`` in ``[0,1]``.
    """
    source = Path(path)
    if str(path).startswith("h5://"):
        from gexco.data.hdf5 import load_view

        array = load_view(str(path), expected_density="diffdrr")
    elif source.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        array = tifffile.imread(source)
    elif source.suffix == ".npy":
        array = np.load(source)
    elif source.suffix in {".pt", ".pth"}:
        loaded = torch.load(source, map_location="cpu", weights_only=True)
        array = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else np.asarray(loaded)
    else:
        with Image.open(source) as image:
            array = np.asarray(image.convert("F"), dtype=np.float32)
    tensor = torch.as_tensor(np.asarray(array).copy(), dtype=torch.float32).squeeze()
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D image at {source}, got {tuple(tensor.shape)}.")
    finite = torch.isfinite(tensor)
    if not finite.all():
        raise ValueError(f"Image contains no finite values: {source}")
    if normalization == "dataset_percentile_fixed":
        if stats_path is None:
            raise ValueError("stats_path is required for checkpoint-exact DRR normalization.")
        statistics = json.loads(Path(stats_path).read_text())
        global_statistics = statistics["train_global"]
        low = tensor.new_tensor(float(global_statistics["p1"]))
        high = tensor.new_tensor(float(global_statistics["p99"]))
        tensor = (tensor.clamp(low, high) - low) / (high - low).clamp_min(1e-6)
    elif normalization == "per_image_minmax":
        low = tensor[finite].amin()
        high = tensor[finite].amax()
        tensor = (tensor - low) / (high - low).clamp_min(1e-6)
    elif normalization != "none":
        raise ValueError("normalization must be 'dataset_percentile_fixed', 'per_image_minmax', or 'none'.")
    tensor = torch.nn.functional.interpolate(
        tensor[None, None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    return tensor.clamp(0.0, 1.0)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON or JSON-lines world-model manifest.

    Args:
        path: Manifest path. JSON may be a list or ``{"samples": [...]}``.

    Returns:
        List of sample dictionaries.
    """
    manifest = Path(path)
    if manifest.suffix == ".jsonl":
        return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    payload = json.loads(manifest.read_text())
    records = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a list of samples.")
    return records


class DrrWorldModelDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample any anchor and ``K`` target angles from one CT's rendered DRRs."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        num_targets: int = 8,
        image_size: int = 128,
        seed: int = 42,
        stats_path: str | Path | None = None,
        normalization: str = "dataset_percentile_fixed",
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest: JSON/JSONL file with ``views`` lists containing paths and angles.
            num_targets: Number of target views ``K``.
            image_size: Output DRR resolution.
            seed: Sampling seed.
            stats_path: Training-set DRR percentile-statistics JSON.
            normalization: DRR normalization policy.
        """
        self.records = read_manifest(manifest)
        self.num_targets = int(num_targets)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.stats_path = None if stats_path is None else Path(stats_path)
        self.normalization = str(normalization)
        self._random = None  # Use DataLoader worker-seeded Python RNG, not copied per-worker state.
        if self.num_targets < 1:
            raise ValueError("num_targets must be positive.")
        if self.normalization == "dataset_percentile_fixed" and self.stats_path is None:
            raise ValueError("stats_path is required for dataset_percentile_fixed normalization.")

    def __len__(self) -> int:
        """Return the number of CT cases."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load an anchor and target set.

        Args:
            index: Dataset index.

        Returns:
            Dictionary containing anchor/target images, angles, and deltas.
        """
        record = self.records[index]
        views = record.get("views", [])
        if len(views) < 2:
            raise ValueError("Each manifest sample needs at least two views.")
        rng = random
        anchor_index = rng.randrange(len(views))
        candidates = [item for item in range(len(views)) if item != anchor_index]
        if len(candidates) >= self.num_targets:
            target_indices = rng.sample(candidates, self.num_targets)
        else:
            raise ValueError("K exceeds available non-anchor views; targets are sampled without replacement")
        anchor_view = views[anchor_index]
        targets = [views[item] for item in target_indices]
        anchor_angle = float(anchor_view["angle_deg"])
        target_angles = torch.tensor([float(view["angle_deg"]) for view in targets])
        deltas = torch.deg2rad(target_angles - anchor_angle)
        return {
            "anchor": load_grayscale_image(
                anchor_view["path"],
                self.image_size,
                stats_path=self.stats_path,
                normalization=self.normalization,
            ),
            "targets": torch.stack(
                [
                    load_grayscale_image(
                        view["path"],
                        self.image_size,
                        stats_path=self.stats_path,
                        normalization=self.normalization,
                    )
                    for view in targets
                ]
            ),
            "anchor_angle": torch.tensor(math.radians(anchor_angle), dtype=torch.float32),
            "target_angles": torch.deg2rad(target_angles),
            "angular_deltas": deltas,
        }


# Public dataset names.
DRRWorldModelDataset = DrrWorldModelDataset
