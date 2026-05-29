"""
Data source: https://zenodo.org/records/12825163
Citation:  Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
"""

import sys
import json
import hashlib
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from neuralop.data.datasets import navier_stokes


PT_TRAIN = "nsforcing_train_128.pt"
PT_TEST  = "nsforcing_test_128.pt"

RESOLUTION = 128

TRAIN_RATIOS: Tuple[float, float, float] = (0.80, 0.10, 0.10)
SEED = 42


def download_dataset(raw_dir: Path) -> None:
  """Trigger neuralop's Zenodo downloader.

  We pass n_train=1/n_tests=[1] — the minimum valid call.
  We only want the .pt files on disk; we build our own DataLoaders
  downstream for full control over splits and normalisation.

  Files written to raw_dir:
    nsforcing_train_128.pt   (~2.4 GB)
    nsforcing_test_128.pt  (~size varies)
  """
  train_present = (raw_dir / PT_TRAIN).exists()
  test_present  = (raw_dir / PT_TEST).exists()

  if train_present and test_present:
    print(f"  -> Both .pt files already present in {raw_dir} (skipping download)")
    return

  print(f"  Downloading NS-2D dataset from Zenodo (record 12825163) ...")
  print(f"  Saving to: {raw_dir}")

  navier_stokes.load_navier_stokes_pt(
    n_train      = 1,
    n_tests      = [1],
    batch_size     = 1,
    test_batch_sizes = [1],
    data_root    = raw_dir,
    train_resolution = RESOLUTION,
    test_resolutions = [RESOLUTION],
    encode_input   = False,
    encode_output  = False,  
    num_workers    = 0,
  )
  print(f"Download complete")


def load_pt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
  print(f"\n  Loading {path.name} ...")
  data = torch.load(path, map_location="cpu", weights_only=True)

  if not isinstance(data, dict) or "x" not in data or "y" not in data:
    raise KeyError(f"Expected dict with keys 'x' and 'y' in {path.name}")

  x = data["x"].numpy().astype(np.float32)  # [N, H, W]
  y = data["y"].numpy().astype(np.float32)  # [N, H, W]

  print(f"  x shape : {x.shape}   y shape : {y.shape}")

  for name, arr in [("x", x), ("y", y)]:
    if np.isnan(arr).any():
      raise ValueError(f"NaN values in '{name}' field of {path.name}")
    if np.isinf(arr).any():
      raise ValueError(f"Inf values in '{name}' field of {path.name}")

  print(f"  x stats : min={x.min():.4f}  max={x.max():.4f}  mean={x.mean():.4f}")
  print(f"  y stats : min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")

  return x, y

def vorticity_to_velocity(omega: np.ndarray) -> np.ndarray:
  H, W = omega.shape[-2:]

  # wavenumbers for a periodic domain of size H×W
  kx = np.fft.fftfreq(W) * W  # [W] — cycles per grid unit
  ky = np.fft.fftfreq(H) * H  # [H]
  KX, KY = np.meshgrid(kx, ky)  # [H, W]
  K2 = KX**2 + KY**2
  K2[0, 0] = 1.0         

  omega_hat      = np.fft.fft2(omega)     # FFT over last two dims
  psi_hat      = -omega_hat / K2       # solve Poisson in Fourier space
  psi_hat[..., 0, 0] = 0.0            # zero-mean gauge fix

  psi = np.fft.ifft2(psi_hat).real        # stream function, real-valued

  u_x =  np.gradient(psi, axis=-2)        # ∂ψ/∂y
  u_y = -np.gradient(psi, axis=-1)        # -∂ψ/∂x

  return np.stack([u_x, u_y], axis=-1)      # [..., H, W, 2]


def process_and_save(
  x: np.ndarray,
  y: np.ndarray,
  processed_dir: Path,
  name: str,
) -> Tuple[Path, np.ndarray, np.ndarray]:

  print(f"  Converting vorticity → velocity ({name}) ...")

  x_vel = vorticity_to_velocity(x)   # [N, H, W, 2]
  y_vel = vorticity_to_velocity(y)   # [N, H, W, 2]

  print(f"x_vel shape : {x_vel.shape}")
  print(f"y_vel shape : {y_vel.shape}")

  processed_dir.mkdir(parents=True, exist_ok=True)
  processed_path = processed_dir / f"nsforcing_{name}_{RESOLUTION}_velocity.npy"

  np.save(processed_path, {"x": x_vel, "y": y_vel}) # type: ignore
  print(f"Saved to {processed_path.name}")

  return processed_path, x_vel, y_vel

def make_splits(
  n: int,
  ratios: Tuple[float, float, float],
  out_path: Path,
  seed: int = SEED,
) -> Dict[str, list]:

  assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios):.6f}"

  train_count = int(n * ratios[0])
  test_count  = int(n * ratios[2])
  val_count   = n - train_count - test_count  

  rng     = np.random.default_rng(seed)
  permutation = rng.permutation(n)

  train_idx = permutation[:train_count].tolist()
  val_idx   = permutation[train_count : train_count + val_count].tolist()
  test_idx  = permutation[train_count + val_count :].tolist()

  assert len(train_idx) + len(val_idx) + len(test_idx) == n

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, train=train_idx, val=val_idx, test=test_idx)

  print(f"  ✓ Splits : train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
  print(f"  ✓ Saved  : {out_path.name}")

  return {"train": train_idx, "val": val_idx, "test": test_idx}


def compute_normalisation_stats(
  processed_path: Path,
  train_indices: List[int],
) -> Dict[str, float]:

  print(f"Computing normalisation stats on {len(train_indices)} train samples ...")

  payload   = np.load(processed_path, allow_pickle=True).item()
  x_train   = payload["x"][train_indices]   # [N_train, H, W, 2]
  y_train   = payload["y"][train_indices]   # [N_train, H, W, 2]

  # u_max over both input and target fields on train split only
  u_max  = float(max(np.abs(x_train).max(), np.abs(y_train).max()))
  u_mean = float((x_train.mean() + y_train.mean()) / 2.0)

  print(f"u_max  : {u_max:.6f}  (normalise all splits by this value)")
  print(f"u_mean : {u_mean:.6f}")

  return {"u_max": u_max, "u_mean": u_mean}


def compute_md5(path: Path) -> str:
  md5 = hashlib.md5()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      md5.update(chunk)
  return md5.hexdigest()


def write_manifest(entries: Dict, dest: Path) -> None:
  dest.parent.mkdir(parents=True, exist_ok=True)
  with open(dest, "w") as f:
    json.dump(entries, f, indent=2)
  print(f"\nManifest written to {dest}")


def check_manifest_valid(manifest_path: Path, raw_dir: Path) -> bool:
  if not manifest_path.exists():
    return False
  try:
    with open(manifest_path) as f:
      manifest = json.load(f)

    for key, filename in [("train", PT_TRAIN), ("test", PT_TEST)]:
      raw_path = raw_dir / filename
      if not raw_path.exists():
        return False
      stored_md5  = manifest.get(key, {}).get("md5")
      current_md5 = compute_md5(raw_path)
      if stored_md5 != current_md5:
        return False

    u_max = manifest.get("train", {}).get("normalisation", {}).get("u_max")
    if not u_max:
      return False

    return True
  except Exception:
    return False


def run_pipeline(args: argparse.Namespace) -> None:
  manifest_output: Dict = {}

  print(f"\n{'='*60}")
  print(f"Step 1 — Download")
  print(f"{'='*60}")
  download_dataset(args.raw_dir)

  print(f"\n{'='*60}")
  print(f"Step 2–4 — Train file")
  print(f"{'='*60}")

  train_pt   = args.raw_dir / PT_TRAIN
  x_tr, y_tr = load_pt(train_pt)
  n_train_total = x_tr.shape[0]  

  train_proc_path, x_vel, y_vel = process_and_save(
    x_tr, y_tr, args.processed_dir, "train"
  )
  del x_tr, y_tr, x_vel, y_vel   

  splits   = make_splits(n_train_total, TRAIN_RATIOS, args.split_dir / "ns_splits.npz")
  norm_stats = compute_normalisation_stats(train_proc_path, splits["train"])

  manifest_output["train"] = {
    "raw_path"    : str(train_pt),
    "processed_path"  : str(train_proc_path),
    "processed_shape" : [n_train_total, RESOLUTION, RESOLUTION, 2],
    "resolution"    : RESOLUTION,
    "md5"       : compute_md5(train_pt),
    "split_file"    : str(args.split_dir / "ns_splits.npz"),
    "split_counts"  : {
      "train": len(splits["train"]),
      "val"  : len(splits["val"]),
      "test" : len(splits["test"]),
    },
    "normalisation"   : norm_stats,
  }

  print(f"\n{'='*60}")
  print(f"Step 5 — Test file (held-out)")
  print(f"{'='*60}")

  test_pt  = args.raw_dir / PT_TEST
  x_te, y_te = load_pt(test_pt)
  n_test_total = x_te.shape[0]

  test_proc_path, _, _ = process_and_save(
    x_te, y_te, args.processed_dir, "test"
  )
  del x_te, y_te

  manifest_output["test"] = {
    "raw_path"    : str(test_pt),
    "processed_path"  : str(test_proc_path),
    "processed_shape" : [n_test_total, RESOLUTION, RESOLUTION, 2],
    "resolution"    : RESOLUTION,
    "md5"       : compute_md5(test_pt),
    "note"      : "held-out set from Zenodo — no further split applied",
  }

  write_manifest(manifest_output, args.manifest_path)

  print(f"\n{'='*60}")
  print(f"Pipeline complete")
  print(f"{'='*60}")
  print(f"  Train samples : {n_train_total}")
  print(f"  train split : {len(splits['train'])}")
  print(f"  val split   : {len(splits['val'])}")
  print(f"  test split  : {len(splits['test'])}")
  print(f"  Test (held-out): {n_test_total}")
  print(f"  u_max       : {norm_stats['u_max']:.6f}")
  print(f"  Manifest    : {args.manifest_path}")
  print(f"\n  Next step: implement data/renderer.py")



if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Download and preprocess NS-2D benchmark dataset"
  )
  parser.add_argument("--raw-dir",     type=Path, default=Path("data/raw"),
            help="where neuralop downloads the .pt files")
  parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
            help="where processed velocity .npy files are saved")
  parser.add_argument("--split-dir",   type=Path, default=Path("data/splits"),
            help="where split index .npz files are saved")
  parser.add_argument("--manifest-path", type=Path, default=Path("data/manifest.json"),
            help="output manifest path")
  parser.add_argument("--force",     action="store_true",
            help="re-run pipeline even if manifest is already valid")
  args = parser.parse_args()

  if not args.force and check_manifest_valid(args.manifest_path, args.raw_dir):
    print("Manifest valid and checksums match. Nothing to do.")
    print("Use --force to re-run the full pipeline.")
    sys.exit(0)

  run_pipeline(args)