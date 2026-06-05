import pytest
import torch
from torch.utils.data import DataLoader

from engine.data.ns_dataset import NavierStokes2DDataset

RES = 16


def test_split_lengths(manifest):
    expected = {"train": 18, "val": 3, "test": 3, "heldout": 6}
    for split, n in expected.items():
        ds = NavierStokes2DDataset(manifest, split=split)
        assert len(ds) == n


def test_velocity_shape_and_normalisation(manifest):
    ds = NavierStokes2DDataset(manifest, split="train", mode="velocity")
    x, y = ds[0]
    assert x.shape == (2, RES, RES) and y.shape == (2, RES, RES)
    assert x.dtype == torch.float32
    assert x.abs().max().item() <= 1.0 + 1e-6


def test_normalize_false_keeps_scale(manifest):
    ds = NavierStokes2DDataset(manifest, split="train", normalize=False)
    assert ds[0][0].abs().max().item() > 1.0


def test_render_mode_channels(manifest):
    from engine.data.renderer import TracerRenderer

    r = TracerRenderer(n_particles=300, n_steps=2, resolution=RES)
    ds = NavierStokes2DDataset(manifest, split="val", mode="render", renderer=r)
    xi, _ = ds[0]
    assert xi.shape == (3, RES, RES)
    assert ds.num_channels == 3


def test_dataloader_collate(manifest):
    ds = NavierStokes2DDataset(manifest, split="train")
    bx, by = next(iter(DataLoader(ds, batch_size=4)))
    assert bx.shape == (4, 2, RES, RES)


@pytest.mark.parametrize("bad", [{"split": "nope"}, {"mode": "nope"}])
def test_invalid_args_raise(manifest, bad):
    with pytest.raises(ValueError):
        NavierStokes2DDataset(manifest, **bad)


def test_oob_index_raises(manifest):
    ds = NavierStokes2DDataset(manifest, split="val")
    with pytest.raises(IndexError):
        _ = ds[len(ds)]