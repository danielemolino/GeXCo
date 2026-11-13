"""SIGReg distributional regularization tests."""

import torch

from gexco.pretraining.sigreg import SIGReg


def test_sigreg_is_non_negative() -> None:
    """SIGReg always returns a non-negative scalar."""
    torch.manual_seed(5)
    loss = SIGReg(knots=17, num_proj=64)(torch.randn(1, 96, 16))
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_collapsed_embeddings_are_penalized_more() -> None:
    """Constant embeddings depart further from a Gaussian than spread embeddings."""
    regularizer = SIGReg(knots=17, num_proj=256)
    torch.manual_seed(7)
    collapsed = regularizer(torch.zeros(1, 256, 32))
    torch.manual_seed(7)
    spread = regularizer(torch.randn(1, 256, 32))
    assert collapsed > spread
