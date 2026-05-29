import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class TracerRenderer:
  def __init__(
    self,
    n_particles: int = 10000,
    dt: float = 0.1,
    n_steps: int = 10,
    resolution: int = 128,
    seed: int = 42,
  ) -> None:
    self.n_particles = n_particles
    self.dt = dt
    self.n_steps = n_steps
    self.resolution = resolution

    rng = torch.Generator()
    rng.manual_seed(seed)
    self.particles = torch.rand((n_particles, 2), generator=rng, dtype=torch.float32)

  def render(self, u_field: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
      rng = torch.Generator()
      rng.manual_seed(seed)
      pos = torch.rand((self.n_particles, 2), generator=rng, dtype=torch.float32)
    else:
      pos = self.particles.clone()

    u_tensor = torch.from_numpy(u_field).float().permute(2, 0, 1).unsqueeze(0)

    for _ in range(self.n_steps):
      grid_pos = pos * 2.0 - 1.0
      grid = grid_pos.unsqueeze(0).unsqueeze(2)
      v_interp = F.grid_sample(u_tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
      v_interp = v_interp.squeeze(0).squeeze(2).t()
      pos = (pos + self.dt * v_interp) % 1.0

    return self._histogram_and_colourise(pos)

  def render_sequence(self, u_sequence: np.ndarray) -> np.ndarray:
    T = u_sequence.shape[0]
    out_images = np.zeros((T, self.resolution, self.resolution, 3), dtype=np.float32)
    pos = self.particles.clone()

    for t in range(T):
      u_tensor = torch.from_numpy(u_sequence[t]).float().permute(2, 0, 1).unsqueeze(0)

      grid_pos = pos * 2.0 - 1.0
      grid = grid_pos.unsqueeze(0).unsqueeze(2)
      v_interp = F.grid_sample(u_tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
      v_interp = v_interp.squeeze(0).squeeze(2).t()
      pos = (pos + self.dt * v_interp) % 1.0

      out_images[t] = self._histogram_and_colourise(pos)

    return out_images

  def _histogram_and_colourise(self, pos: torch.Tensor) -> np.ndarray:
    pos_np = pos.numpy()
    hist, _, _ = np.histogram2d(
      pos_np[:, 1], pos_np[:, 0],
      bins=[self.resolution, self.resolution],
      range=[[0.0, 1.0], [0.0, 1.0]],
    )
    if hist.max() > 0:
      hist = hist / hist.max()
    hist = np.clip(hist, 0.0, 1.0).astype(np.float32)
    return self._colourise(hist)

  def _colourise(self, density: np.ndarray) -> np.ndarray:
    low  = np.array([0.05, 0.10, 0.30], dtype=np.float32)
    high = np.array([0.20, 0.85, 0.95], dtype=np.float32)
    return low + density[..., None] * (high - low)