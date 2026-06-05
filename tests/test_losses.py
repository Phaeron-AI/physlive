import pytest
import torch

from engine.training.losses import H1Loss, RelativeLpLoss, build_loss


def test_relative_l2_zero_on_identical():
    t = torch.randn(4, 2, 16, 16)
    assert RelativeLpLoss()(t, t).item() == pytest.approx(0.0, abs=1e-6)


def test_relative_l2_scale_invariant():
    pred, target = torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8)
    loss = RelativeLpLoss()
    base = loss(pred, target).item()
    assert loss(3.0 * pred, 3.0 * target).item() == pytest.approx(base, rel=1e-5)


def test_relative_l2_known_value():
    # pred = 0 -> ||0-target|| / ||target|| == 1 for every sample
    target = torch.randn(5, 2, 8, 8)
    assert RelativeLpLoss()(torch.zeros_like(target), target).item() == pytest.approx(1.0, rel=1e-5)


def test_h1_zero_on_identical_and_ge_l2():
    t = torch.randn(3, 2, 16, 16)
    assert H1Loss()(t, t).item() == pytest.approx(0.0, abs=1e-6)
    pred = t + 0.1 * torch.randn_like(t)
    assert H1Loss(alpha=1.0)(pred, t).item() >= RelativeLpLoss()(pred, t).item()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        RelativeLpLoss()(torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 9))


def test_build_loss_dispatch():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"name": "h1", "alpha": 0.5})
    assert isinstance(build_loss(cfg), H1Loss)
    with pytest.raises(KeyError):
        build_loss(OmegaConf.create({"name": "nope"}))