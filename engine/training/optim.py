from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

__all__ = ["build_optimizer", "build_scheduler"]

_OPTIMIZERS = {
  "adam": torch.optim.Adam,
  "adamw": torch.optim.AdamW,
  "sgd": torch.optim.SGD,
}


def build_optimizer(cfg: Any, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
  name = str(cfg.name).lower()
  if name not in _OPTIMIZERS:
    raise KeyError(f"unknown optimizer {name!r}; available: {sorted(_OPTIMIZERS)}")
  kwargs = {k: v for k, v in dict(cfg).items() if k != "name"}
  return _OPTIMIZERS[name](params, **kwargs)


def build_scheduler(cfg: Optional[Any], optimizer: torch.optim.Optimizer):
  if cfg is None or str(getattr(cfg, "name", "none")).lower() == "none":
    return None
  name = str(cfg.name).lower()
  kwargs = {k: v for k, v in dict(cfg).items() if k != "name"}

  if name == "step":
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
  if name == "cosine":
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
  if name == "plateau":
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
  raise KeyError(f"unknown scheduler {name!r}")