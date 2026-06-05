from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["RelativeLpLoss", "H1Loss", "build_loss"]


def _flatten_per_sample(t: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, C*H*W]
    return t.reshape(t.shape[0], -1)


class RelativeLpLoss(nn.Module):
  def __init__(self, p: float = 2.0, eps: float = 1e-8) -> None:
    super().__init__()
    if p <= 0:
      raise ValueError(f"p must be positive, got {p}")
    self.p = float(p)
    self.eps = float(eps)

  def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
      raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    diff = _flatten_per_sample(pred - target).norm(p=self.p, dim=1)
    denom = _flatten_per_sample(target).norm(p=self.p, dim=1).clamp_min(self.eps)
    return (diff / denom).mean()


class H1Loss(nn.Module):
  def __init__(self, alpha: float = 1.0, eps: float = 1e-8) -> None:
    super().__init__()
    self.alpha = float(alpha)
    self.base = RelativeLpLoss(p=2.0, eps=eps)

  @staticmethod
  def _grads(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = t[..., 1:, :] - t[..., :-1, :]
    dy = t[..., :, 1:] - t[..., :, :-1]
    return dx, dy

  def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    value = self.base(pred, target)
    px, py = self._grads(pred)
    tx, ty = self._grads(target)
    grad = self.base(px, tx) + self.base(py, ty)
    return value + self.alpha * grad


_LOSSES = {
  "l2": RelativeLpLoss,          
  "relative_lp": RelativeLpLoss,
  "h1": H1Loss,
  "mse": nn.MSELoss,
}


def build_loss(cfg: Any) -> nn.Module:
  from ..utils.config import call_filtered

  name = str(cfg.name).lower()
  if name not in _LOSSES:
    raise KeyError(f"unknown loss {name!r}; available: {sorted(_LOSSES)}")
  params = {k: v for k, v in dict(cfg).items() if k != "name"}
  return call_filtered(_LOSSES[name], params)