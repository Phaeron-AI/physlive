"""DataLoader construction from config."""

from __future__ import annotations

from typing import Any, Optional

from torch.utils.data import DataLoader

from .ns_dataset import NavierStokes2DDataset

__all__ = ["build_dataset", "build_loader"]


def build_dataset(cfg: Any, split: str) -> NavierStokes2DDataset:
  return NavierStokes2DDataset(
    manifest_path=cfg.manifest_path,
    split=split,
    mode=getattr(cfg, "mode", "velocity"),
    normalize=getattr(cfg, "normalize", True),
  )


def build_loader(cfg: Any, split: str, *, shuffle: Optional[bool] = None) -> DataLoader:
  dataset = build_dataset(cfg, split)
  if shuffle is None:
    shuffle = split == "train"
  return DataLoader(
    dataset,
    batch_size=int(cfg.batch_size),
    shuffle=shuffle,
    num_workers=int(getattr(cfg, "num_workers", 0)),
    pin_memory=bool(getattr(cfg, "pin_memory", False)),
    drop_last=bool(getattr(cfg, "drop_last", False)) and shuffle,
    persistent_workers=bool(getattr(cfg, "num_workers", 0)) > 0,
  )