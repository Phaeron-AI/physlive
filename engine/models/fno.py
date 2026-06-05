from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn

__all__ = ["build_model", "register_model", "build_fno"]

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:

  def _decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
    key = name.lower()
    if key in _REGISTRY:
      raise KeyError(f"model {name!r} already registered")
    _REGISTRY[key] = fn
    return fn

  return _decorator


def build_model(cfg: Any) -> nn.Module:
  name = str(cfg.name).lower()
  if name not in _REGISTRY:
    raise KeyError(
      f"unknown model {name!r}; registered: {sorted(_REGISTRY)}"
    )
  params = {k: v for k, v in dict(cfg).items() if k != "name"}
  return _REGISTRY[name](**params)


@register_model("fno")
def build_fno(
    *,
    n_modes: tuple[int, int] = (16, 16),
    in_channels: int = 2,
    out_channels: int = 2,
    hidden_channels: int = 64,
    n_layers: int = 4,
    **extra: Any,
) -> nn.Module:

  from neuralop.models import FNO  

  n_modes = tuple(n_modes)  # type: ignore[assignment]
  return FNO(
    n_modes=n_modes,
    in_channels=in_channels,
    out_channels=out_channels,
    hidden_channels=hidden_channels,
    n_layers=n_layers,
    **extra,
  )