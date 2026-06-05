from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:  # avoid importing torch-heavy renderer unless render mode is used
    from .renderer import TracerRenderer

__all__ = ["NavierStokes2DDataset"]

# Public split names accepted by the dataset.
_FILE_SPLITS = ("train", "val", "test")  # index into the train processed file
_HELDOUT = "heldout"                      # the separate Zenodo test file
VALID_SPLITS = (*_FILE_SPLITS, _HELDOUT)

VALID_MODES = ("velocity", "render")

Sample = Tuple[torch.Tensor, torch.Tensor]
Transform = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def _resolve_renderer(resolution: int) -> "TracerRenderer":
    """Construct a default renderer, tolerating both package and script imports."""
    try:
        from .renderer import TracerRenderer
    except ImportError:  # pragma: no cover - fallback when run outside the package
        from renderer import TracerRenderer  # type: ignore[no-redef]
    return TracerRenderer(resolution=resolution)


class NavierStokes2DDataset(Dataset):
    """Lazily-served Navier-Stokes 2D velocity-field dataset.

    The full processed array for the relevant file is memory-resident once per
    instance (the upstream pickled-dict ``.npy`` format cannot be memory-mapped
    per-key). Items are *indexed* out of that array on access rather than copied
    up front, so the array can be shared copy-on-write across DataLoader workers
    on fork-based platforms.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.json`` written by the download pipeline.
    split:
        One of ``"train"``, ``"val"``, ``"test"`` (rows of the train processed
        file selected via ``ns_splits.npz``) or ``"heldout"`` (the entire Zenodo
        test processed file).
    mode:
        ``"velocity"`` to return normalised velocity tensors, or ``"render"`` to
        return tracer-density RGB images.
    normalize:
        Whether to divide velocity fields by ``u_max`` (velocity mode only).
    renderer:
        Optional :class:`TracerRenderer` instance. If ``None`` and ``mode`` is
        ``"render"``, a default renderer is created at the dataset resolution.
    cache_rendered:
        Cache rendered images in memory after first access. Trades memory for
        speed across epochs; only meaningful in render mode.
    transform:
        Optional callable applied to ``(x, y)`` after mode-specific processing.
    dtype:
        Output floating dtype (default ``torch.float32``).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        *,
        mode: str = "velocity",
        normalize: bool = True,
        renderer: Optional["TracerRenderer"] = None,
        cache_rendered: bool = False,
        transform: Optional[Transform] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS!r}, got {split!r}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES!r}, got {mode!r}")

        self.split = split
        self.mode = mode
        self.normalize = normalize
        self.transform = transform
        self.dtype = dtype

        manifest_path = Path(manifest_path)
        manifest = self._load_manifest(manifest_path)
        manifest_dir = manifest_path.parent

        # Normalisation is always taken from the train split statistics so that
        # every split is scaled by the same constant.
        self.u_max = float(_dig(manifest, "train", "normalisation", "u_max"))
        if self.u_max <= 0.0:
            raise ValueError(f"manifest u_max must be positive, got {self.u_max}")

        processed_path, indices = self._resolve_source(manifest, manifest_dir)
        self.resolution = int(_dig(manifest, self._manifest_key(), "resolution"))

        self._x, self._y = self._load_fields(processed_path)
        self._indices = indices

        # Validate index range against the loaded array up front for fast failure.
        n = self._x.shape[0]
        if self._indices.size and (self._indices.min() < 0 or self._indices.max() >= n):
            raise IndexError(
                f"split {split!r} references indices outside the processed array "
                f"of length {n}"
            )

        self._renderer = renderer
        self._cache_rendered = cache_rendered
        self._render_cache: Dict[int, Sample] = {}
        if self.mode == "render" and self._renderer is None:
            self._renderer = _resolve_renderer(self.resolution)

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} out of range for split of size {len(self)}")

        row = int(self._indices[idx])

        if self.mode == "render":
            sample = self._get_rendered(idx, row)
        else:
            sample = self._get_velocity(row)

        if self.transform is not None:
            sample = self.transform(*sample)
        return sample

    # ------------------------------------------------------------------ #
    # Item construction
    # ------------------------------------------------------------------ #
    def _get_velocity(self, row: int) -> Sample:
        x = self._to_chw(self._x[row])
        y = self._to_chw(self._y[row])
        if self.normalize:
            x = x / self.u_max
            y = y / self.u_max
        return x, y

    def _get_rendered(self, idx: int, row: int) -> Sample:
        if self._cache_rendered and idx in self._render_cache:
            return self._render_cache[idx]

        assert self._renderer is not None  # established in __init__ for render mode
        # Render from raw velocity: advection magnitude is physically meaningful.
        x_img = self._renderer.render(np.ascontiguousarray(self._x[row]))
        y_img = self._renderer.render(np.ascontiguousarray(self._y[row]))
        sample = (self._to_chw(x_img), self._to_chw(y_img))

        if self._cache_rendered:
            self._render_cache[idx] = sample
        return sample

    def _to_chw(self, field: np.ndarray) -> torch.Tensor:
        # [H, W, C] -> [C, H, W]
        return torch.from_numpy(field).permute(2, 0, 1).contiguous().to(self.dtype)

    # ------------------------------------------------------------------ #
    # Loading / manifest resolution
    # ------------------------------------------------------------------ #
    def _manifest_key(self) -> str:
        return "test" if self.split == _HELDOUT else "train"

    def _resolve_source(
        self, manifest: Dict[str, Any], manifest_dir: Path
    ) -> Tuple[Path, np.ndarray]:
        key = self._manifest_key()
        processed_path = self._resolve_path(
            _dig(manifest, key, "processed_path"), manifest_dir
        )

        if self.split == _HELDOUT:
            n = int(_dig(manifest, "test", "processed_shape")[0])
            return processed_path, np.arange(n, dtype=np.int64)

        split_file = self._resolve_path(
            _dig(manifest, "train", "split_file"), manifest_dir
        )
        if not split_file.exists():
            raise FileNotFoundError(f"split file not found: {split_file}")
        with np.load(split_file) as splits:
            if self.split not in splits:
                raise KeyError(
                    f"split {self.split!r} not present in {split_file.name} "
                    f"(found {list(splits.files)})"
                )
            return processed_path, np.asarray(splits[self.split], dtype=np.int64)

    @staticmethod
    def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest not found: {manifest_path}. Run data/download_ns.py first."
            )
        with open(manifest_path) as f:
            return json.load(f)

    @staticmethod
    def _resolve_path(path_str: str, manifest_dir: Path) -> Path:
        """Resolve a manifest path, falling back to manifest-relative if needed.

        Upstream stores paths as given on the command line (often relative to the
        run directory). If such a path does not exist as-is, retry it relative to
        the manifest's own directory, which is robust to where the job was run.
        """
        path = Path(path_str)
        if path.exists():
            return path
        candidate = manifest_dir / path.name
        if candidate.exists():
            return candidate
        # Return the original; downstream load raises a clear FileNotFoundError.
        return path

    @staticmethod
    def _load_fields(processed_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not processed_path.exists():
            raise FileNotFoundError(f"processed file not found: {processed_path}")
        payload = np.load(processed_path, allow_pickle=True).item()
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            raise KeyError(
                f"expected dict with keys 'x' and 'y' in {processed_path.name}"
            )
        x = np.ascontiguousarray(payload["x"], dtype=np.float32)
        y = np.ascontiguousarray(payload["y"], dtype=np.float32)
        if x.ndim != 4 or x.shape[-1] != 2 or x.shape != y.shape:
            raise ValueError(
                f"expected x and y of shape [N, H, W, 2] in {processed_path.name}, "
                f"got x={x.shape} y={y.shape}"
            )
        return x, y

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def num_channels(self) -> int:
        return 3 if self.mode == "render" else 2

    @property
    def indices(self) -> List[int]:
        return self._indices.tolist()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(split={self.split!r}, mode={self.mode!r}, "
            f"n={len(self)}, resolution={self.resolution}, "
            f"u_max={self.u_max:.4f}, normalize={self.normalize})"
        )


def _dig(mapping: Dict[str, Any], *keys: str) -> Any:
    """Traverse nested dict keys, raising a clear error on the first miss."""
    node: Any = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            trail = " -> ".join(keys)
            raise KeyError(f"manifest missing expected entry: {trail}")
        node = node[key]
    return node