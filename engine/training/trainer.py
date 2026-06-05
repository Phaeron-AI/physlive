"""Training loop, checkpointing, and resume logic.

A dependency-light ``Trainer`` (no Lightning) that owns the epoch loop, mixed
precision, gradient clipping, scheduler stepping, metric tracking, and
checkpoint save/load. It accepts any ``nn.Module`` so it can be smoke-tested
with a trivial model independently of the FNO/neuralop stack.

Distributed (DDP) training is intentionally out of scope here; the loop is
structured so a DistributedSampler + DDP wrap can be added without reshaping it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

__all__ = ["Trainer", "TrainState"]


@dataclass
class TrainState:
    """Mutable training state, persisted in checkpoints for clean resume."""

    epoch: int = 0
    global_step: int = 0
    best_val: float = float("inf")
    history: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        *,
        scheduler: Optional[Any] = None,
        device: str | torch.device = "cpu",
        ckpt_dir: str | Path = "checkpoints",
        amp: bool = False,
        grad_clip: Optional[float] = None,
        log_every: int = 50,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.log_every = max(1, int(log_every))

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # AMP only meaningful on CUDA.
        self.amp = bool(amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.amp)

        self.state = TrainState()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
    ) -> TrainState:
        start = self.state.epoch
        for epoch in range(start, epochs):
            self.state.epoch = epoch
            t0 = time.perf_counter()

            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = (
                self._run_epoch(val_loader, train=False)
                if val_loader is not None
                else float("nan")
            )

            if self.scheduler is not None:
                self._step_scheduler(val_loss)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - t0,
            }
            self.state.history.append(row)
            logger.info(
                "epoch %d | train %.6f | val %.6f | lr %.2e | %.1fs",
                epoch, train_loss, val_loss, row["lr"], row["seconds"],
            )

            self.save_checkpoint(self.ckpt_dir / "last.pt")
            if val_loader is not None and val_loss < self.state.best_val:
                self.state.best_val = val_loss
                self.save_checkpoint(self.ckpt_dir / "best.pt")
                logger.info("new best val %.6f -> best.pt", val_loss)

        return self.state

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        return self._run_epoch(loader, train=False)

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #
    def _run_epoch(self, loader: DataLoader, *, train: bool) -> float:
        self.model.train(train)
        total, count = 0.0, 0

        for i, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            bs = x.shape[0]

            with torch.set_grad_enabled(train), torch.autocast(
                device_type=self.device.type, enabled=self.amp
            ):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

            if train:
                self._backward_step(loss)
                self.state.global_step += 1
                if self.state.global_step % self.log_every == 0:
                    logger.debug("step %d | loss %.6f", self.state.global_step, loss.item())

            total += loss.item() * bs
            count += bs

        return total / max(1, count)

    def _backward_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _step_scheduler(self, val_loss: float) -> None:
        # ReduceLROnPlateau needs a metric; others step blind.
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "state": vars(self.state),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, *, weights_only_model: bool = False) -> None:
        # These are first-party checkpoints we wrote (they carry non-tensor
        # training state), so full unpickling is intended and safe here.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if weights_only_model:
            return
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.state = TrainState(**ckpt["state"])
        # resume on the next epoch after the one we saved
        self.state.epoch += 1