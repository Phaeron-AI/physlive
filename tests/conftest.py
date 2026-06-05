"""Shared fixtures: build pipeline-shaped artifacts in a temp dir."""

import json
from pathlib import Path

import numpy as np
import pytest

RES = 16


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    """Create synthetic processed files, splits, and manifest; return its path."""
    rng = np.random.default_rng(0)
    n_train_file, n_test_file = 24, 6

    def fields(n):
        return (rng.standard_normal((n, RES, RES, 2)) * 3.0).astype(np.float32)

    x_tr, y_tr = fields(n_train_file), fields(n_train_file)
    x_te, y_te = fields(n_test_file), fields(n_test_file)

    proc = tmp_path / "processed"
    proc.mkdir()
    train_proc = proc / f"nsforcing_train_{RES}_velocity.npy"
    test_proc = proc / f"nsforcing_test_{RES}_velocity.npy"
    np.save(train_proc, {"x": x_tr, "y": y_tr})
    np.save(test_proc, {"x": x_te, "y": y_te})

    perm = np.random.default_rng(42).permutation(n_train_file)
    tr, va, te = perm[:18].tolist(), perm[18:21].tolist(), perm[21:].tolist()
    split_file = tmp_path / "splits" / "ns_splits.npz"
    split_file.parent.mkdir()
    np.savez(split_file, train=tr, val=va, test=te)

    u_max = float(max(np.abs(x_tr[np.array(tr)]).max(), np.abs(y_tr[np.array(tr)]).max()))

    manifest = {
        "train": {
            "processed_path": str(train_proc),
            "processed_shape": [n_train_file, RES, RES, 2],
            "resolution": RES,
            "split_file": str(split_file),
            "split_counts": {"train": len(tr), "val": len(va), "test": len(te)},
            "normalisation": {"u_max": u_max, "u_mean": 0.0},
        },
        "test": {
            "processed_path": str(test_proc),
            "processed_shape": [n_test_file, RES, RES, 2],
            "resolution": RES,
            "note": "held-out",
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path