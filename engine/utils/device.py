"""Runtime/device helpers."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["resolve_device"]


def resolve_device(requested: str) -> str:
  if requested == "cuda" and not torch.cuda.is_available():
    logger.warning("cuda requested but unavailable; falling back to cpu")
    return "cpu"
  return requested