import os
from pathlib import Path
from dotenv import load_dotenv

from neuralop.data.datasets import navier_stokes

load_dotenv(".env.local")

SAVE_PATH = str(os.getenv("SAVE_PATH_FNO"))

def download_via_neuralop(data_root: str) -> None:
    print("  Downloading via neuralop (Zenodo record 12825163) ...")
    navier_stokes.load_navier_stokes_pt(
      n_train=1,        
      n_tests=[1],
      batch_size=1,
      test_batch_sizes=[1],
      data_root=Path(data_root),
      train_resolution=128,
      test_resolutions=[128],
      encode_input=False,
      encode_output=False,
      num_workers=0,
    )
    print("Raw .pt files present at", data_root)

# download_via_neuralop(SAVE_PATH)

import torch
data = torch.load(r"D:\Phaeron\physlive\engine\data\raw\nsforcing_train_128.pt")
print(type(data))
print(data.shape if hasattr(data, 'shape') else {k: v.shape for k, v in data.items()})