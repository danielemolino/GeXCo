"""Unified manifest datasets for CT-RATE, LIDC-IDRI, and RadChestCT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gexco.data.preprocessing import preprocess_volume
from gexco.pretraining.data import load_grayscale_image


def load_volume(path: str | Path | dict[str, Any]) -> torch.Tensor:
    """Load a volume from NumPy, PyTorch, NIfTI, or an indexed HDF5 latent.

    Args:
        path: Volume path or ``{"hdf5": path, "row": integer}`` latent reference.

    Returns:
        Float tensor shaped ``(C,D,H,W)``. Scalar CT volumes receive one channel.
    """
    hdf5_reference = isinstance(path, dict)
    if hdf5_reference:
        hdf5_path = path.get("hdf5", path.get("path"))
        row = path.get("row")
        if hdf5_path is None or row is None:
            raise ValueError("An HDF5 volume reference requires 'hdf5' and 'row'.")
        try:
            import h5py
        except ImportError as error:
            raise ImportError("Install the pinned GeXCo dependencies to load HDF5 latents.") from error
        with h5py.File(hdf5_path, "r") as h5:
            value: Any = h5["latents"][int(row)]
        source = Path(hdf5_path)
    else:
        source = Path(path)
    if hdf5_reference:
        pass
    elif source.suffix == ".npy":
        value: Any = np.load(source)
    elif source.suffix in {".pt", ".pth"}:
        value = torch.load(source, map_location="cpu", weights_only=True)
    elif source.name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as error:
            raise ImportError("Install the pinned GeXCo dependencies to load NIfTI volumes.") from error
        value = np.asarray(nib.load(str(source)).get_fdata(dtype=np.float32))
    else:
        raise ValueError(f"Unsupported volume format: {source}")
    tensor = value.float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    tensor = tensor.squeeze()
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        if tensor.shape[-1] == 4 and tensor.shape[0] != 4:
            tensor = tensor.permute(3, 0, 1, 2).contiguous()
        return tensor
    raise ValueError(f"Expected a 3D volume or channel-first latent at {source}, got {tuple(tensor.shape)}.")


class ReconstructionDataset(Dataset[dict[str, Any]]):
    """Manifest-backed biplanar DRR and CT dataset."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_size: int = 128,
        volume_size: int = 128,
        preprocess_ct: bool = True,
        drr_stats_path: str | Path | None = None,
        drr_normalization: str = "dataset_percentile_fixed",
        drr_density_mode: str = "physical_mu",
    ) -> None:
        """Initialize a reconstruction dataset.

        Args:
            manifest: JSON/JSONL entries with ``volume`` and two ``views``.
            image_size: DRR resolution.
            volume_size: CT cube edge.
            preprocess_ct: Apply HU clipping and resampling.
            drr_stats_path: Training-set DRR percentile-statistics JSON.
            drr_normalization: DRR normalization policy.
        """
        source = Path(manifest)
        self.manifest_dir = source.parent
        if source.suffix == ".jsonl":
            self.records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(source.read_text())
            self.records = payload.get("samples", payload) if isinstance(payload, dict) else payload
        if not isinstance(self.records, list):
            raise ValueError("Manifest must contain a list of samples.")
        self.image_size = int(image_size)
        self.volume_size = int(volume_size)
        self.preprocess_ct = bool(preprocess_ct)
        self.drr_stats_path = None if drr_stats_path is None else Path(drr_stats_path)
        self.drr_normalization = str(drr_normalization)
        self.drr_density_mode = str(drr_density_mode)
        if self.drr_normalization == "dataset_percentile_fixed" and self.drr_stats_path is None:
            raise ValueError("drr_stats_path is required for checkpoint-exact normalization.")

    def __len__(self) -> int:
        """Return the number of manifest samples."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one CT and its radiographic views.

        Args:
            index: Dataset index.

        Returns:
            Dictionary with ``case_id``, ``volume``, ``views``, and ``angles``.
        """
        record = self.records[index]
        volume = load_volume(record["volume"])
        if self.preprocess_ct:
            volume = preprocess_volume(volume, output_size=self.volume_size)
        views = sorted(record.get("views", []), key=lambda item: float(item["angle_deg"]))
        if [float(v["angle_deg"]) for v in views] != [0.0, 90.0]:
            raise ValueError("Generative model requires ordered 0-degree frontal and 90-degree lateral views")
        if not views:
            raise ValueError("A reconstruction sample must contain at least one view.")
        sample = {
            "case_id": str(record.get("case_id", index)),
            "volume": volume,
            "views": torch.stack(
                [
                    load_grayscale_image(
                        item["path"],
                        self.image_size,
                        stats_path=self.drr_stats_path,
                        normalization=self.drr_normalization,
                        expected_density=self.drr_density_mode,
                    )
                    for item in views
                ]
            ),
            "angles": torch.tensor([float(item["angle_deg"]) for item in views]),
        }
        spacing_value = record.get("spacing_mm", record.get("spacing"))
        if spacing_value is not None:
            if isinstance(spacing_value, (str, Path)):
                spacing_payload = json.loads(Path(spacing_value).read_text())
                spacing_value = spacing_payload.get("spacing_mm", spacing_payload.get("spacing"))
            if spacing_value is None or len(spacing_value) != 3:
                raise ValueError("Per-volume spacing must contain three z-y-x values.")
            sample["spacing_mm"] = torch.tensor(spacing_value, dtype=torch.float32)
        if record.get("latent"):
            latent_reference = record["latent"]
            if isinstance(latent_reference, dict):
                latent_reference = dict(latent_reference)
                key = "hdf5" if "hdf5" in latent_reference else "path"
                latent_path = Path(latent_reference[key])
                if not latent_path.is_absolute():
                    latent_reference[key] = str(self.manifest_dir / latent_path)
            sample["latent"] = load_volume(latent_reference)
        return sample


class CTRateDataset(ReconstructionDataset):
    """CT-RATE specialization of the common reconstruction manifest."""


class LIDCIDRIDataset(ReconstructionDataset):
    """LIDC-IDRI specialization of the common reconstruction manifest."""


class RadChestCTDataset(ReconstructionDataset):
    """RadChestCT specialization of the common reconstruction manifest."""
