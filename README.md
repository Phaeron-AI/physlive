# PhysLive

Physics-informed live photo generation.  
A two-stage model that estimates physical fields from still images  
and renders temporally coherent, physically consistent animations.

---

## Getting started

```bash
git clone https://github.com/Phaeron-AI/physlive.git
cd physlive
pip install -r requirements.txt

# Week 1: download and verify data
python scripts/download_ns.py --data-dir data/raw

# Run tests (all should fail with NotImplementedError until you implement)
pytest tests/ -v

# Once implemented, all tests should pass before any training
pytest tests/ -v --tb=short
```

## Repository layout

```
phlux/
├── configs/        # all hyperparameters — nothing hardcoded in Python
├── data/           # dataset, renderer, transforms
├── tests/          # shape/dtype assertions — run before every experiment
└── scripts/        # download, preprocess, evaluate
```

## Week-by-week targets

| Week | Target | Done? |
|------|--------|-------|
| 1 | `download_ns.py` runs, manifest written, DataLoader yields correct shapes | |
| 2 | `renderer.py` renders plausible fluid images from NS fields | |
| 3 | Real video scraper + RAFT optical flow pseudo-labels | |
| 4 | Vanilla DiT baseline trains, eval harness reports div-free error | |

## Tensor contract

All tensors follow this convention throughout the codebase.  
Deviating from this convention requires a comment explaining why.

| Name | Shape | Dtype | Range |
|------|-------|-------|-------|
| `image` | `[3, H, W]` | float32 | normalised |
| `u_field` | `[T, H, W, 2]` | float32 | normalised to [-1, 1] |
| `p_field` | `[T, H, W, 1]` | float32 | normalised |

**Time-first convention**: `[T, H, W, C]` throughout.  
The raw `.mat` files are time-last `[N, H, W, T]` — permuted once in `download_ns.py`.

## Physics units

`u_max` is stored in `data/manifest.json` and loaded by every dataset.  
PDE residual losses are always computed in physical units (after denormalising).  
Never compute physics losses on normalised fields.