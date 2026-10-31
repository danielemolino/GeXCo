"""Sanity tests for volume and segmentation metrics."""

from pathlib import Path

import numpy as np
import torch

from gexco.data.preprocessing import clip_and_normalize_hu, denormalize_hu
from gexco.evaluation.run_suite import evaluate_mask_directories
from gexco.evaluation.segmentation import dice_score
from gexco.evaluation.volume_metrics import peak_signal_noise_ratio, structural_similarity


def test_identical_volume_metrics(dummy_volume: torch.Tensor) -> None:
    """Identical volumes have infinite PSNR and exactly unit SSIM."""
    assert peak_signal_noise_ratio(dummy_volume, dummy_volume) > 100.0
    assert structural_similarity(dummy_volume, dummy_volume) == 1.0


def test_identical_masks_have_unit_dice() -> None:
    """Identical non-empty masks have Dice equal to one."""
    mask = torch.zeros(16, 16, 16, dtype=torch.bool)
    mask[3:11, 4:12, 2:9] = True
    assert dice_score(mask, mask) == 1.0


def test_hu_normalization_round_trip() -> None:
    """The MAISI intensity contract maps clipped HU to [0,1] and back."""
    hu = torch.tensor([-1200.0, -1000.0, 0.0, 1000.0, 1200.0])
    normalized = clip_and_normalize_hu(hu)
    assert torch.allclose(
        normalized,
        torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
    )
    assert torch.allclose(
        denormalize_hu(normalized),
        torch.tensor([-1000.0, -1000.0, 0.0, 1000.0, 1000.0]),
    )


def test_segmentation_evaluation_respects_case_limit(tmp_path: Path) -> None:
    """Evaluation uses every case by default and limits a sorted subset on request."""
    predictions = tmp_path / "predictions"
    targets = tmp_path / "targets"
    predictions.mkdir()
    targets.mkdir()
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    for case_id in ("case_c", "case_a", "case_b"):
        np.save(predictions / f"{case_id}.npy", mask)
        np.save(targets / f"{case_id}.npy", mask)

    complete = evaluate_mask_directories(predictions, targets)
    limited = evaluate_mask_directories(predictions, targets, max_cases=2)

    assert complete["summary"]["num_cases"] == 3
    assert limited["summary"]["num_cases"] == 2
    assert [case["case_id"] for case in limited["cases"]] == ["case_a", "case_b"]
