"""Trainer spine test: tiny conv model on synthetic data, no neuralop needed."""

import torch
import torch.nn as nn

from engine.data.loaders import build_loader
from engine.training.losses import RelativeLpLoss
from engine.training.trainer import Trainer
from omegaconf import OmegaConf


class TinyOperator(nn.Module):
    """Channel-preserving conv stack: [B,2,H,W] -> [B,2,H,W]."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 2, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


def _data_cfg(manifest):
    return OmegaConf.create(
        {"manifest_path": str(manifest), "batch_size": 4, "num_workers": 0}
    )


def test_trainer_fit_and_checkpoint(manifest, tmp_path):
    cfg = _data_cfg(manifest)
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)

    model = TinyOperator()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt, RelativeLpLoss(), ckpt_dir=tmp_path / "ckpt", log_every=1)

    state = trainer.fit(train_loader, val_loader, epochs=3)

    assert len(state.history) == 3
    assert (tmp_path / "ckpt" / "last.pt").exists()
    assert (tmp_path / "ckpt" / "best.pt").exists()
    # loss should be finite and the model should learn *something* on fixed data
    assert all(torch.isfinite(torch.tensor(r["train_loss"])) for r in state.history)
    assert state.history[-1]["train_loss"] < state.history[0]["train_loss"]


def test_resume_continues_epoch(manifest, tmp_path):
    cfg = _data_cfg(manifest)
    loader = build_loader(cfg, "train", shuffle=False)

    model = TinyOperator()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t1 = Trainer(model, opt, RelativeLpLoss(), ckpt_dir=tmp_path / "ckpt")
    t1.fit(loader, None, epochs=2)

    model2 = TinyOperator()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    t2 = Trainer(model2, opt2, RelativeLpLoss(), ckpt_dir=tmp_path / "ckpt")
    t2.load_checkpoint(tmp_path / "ckpt" / "last.pt")
    assert t2.state.epoch == 2  # resumes on the epoch after the saved one