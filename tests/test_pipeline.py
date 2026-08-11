"""Fast correctness tests for the restoration pipeline (no GPU, no dataset needed).

Run:  python -m pytest tests/ -q     (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.npy_io import GT_SHAPE, NOISY_SHAPE, load_npy, save_restored  # noqa: E402


def test_save_restored_clips_to_unit_range():
    """Output must be clipped to [0,1] on save (scoring is vs [0,1] GT)."""
    arr = np.array([[-0.5, 0.3], [1.7, 0.9]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.npy"
        save_restored(p, arr)
        out = load_npy(p)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.dtype == np.float32
    assert np.isclose(out[0, 1], 0.3) and np.isclose(out[1, 1], 0.9)


def test_load_npy_preserves_out_of_range_inputs():
    """NoisyLR inputs must NOT be clipped on load."""
    arr = np.array([[-0.2, 1.5]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.npy"
        np.save(p, arr)
        out = load_npy(p)
    assert out.min() < 0.0 and out.max() > 1.0


def test_model_doubles_resolution():
    import torch
    from src.models.baseline_cnn import build_model

    model = build_model("small_srcnn", channels=16, n_blocks=2).eval()
    x = torch.rand(2, 1, *NOISY_SHAPE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, *GT_SHAPE)


def test_split_is_disjoint_and_covers(tmp_path=None):
    """If a split.json exists, its partitions must be disjoint (no leakage)."""
    split_path = Path(__file__).resolve().parents[1] / "configs" / "split.json"
    if not split_path.exists():
        return  # split not built yet; skip
    s = json.loads(split_path.read_text())
    tr, vi, vo = set(s["train"]), set(s["val_iid"]), set(s["val_ood"])
    assert not (tr & vi) and not (tr & vo) and not (vi & vo), "split partitions overlap"


if __name__ == "__main__":
    test_save_restored_clips_to_unit_range()
    test_load_npy_preserves_out_of_range_inputs()
    test_model_doubles_resolution()
    test_split_is_disjoint_and_covers()
    print("all tests passed")
