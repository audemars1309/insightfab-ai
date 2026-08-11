"""Tests for the Phase 9 synthetic degradation simulator.

Covers: output dimensions, value handling (dtype, out-of-range, clipping),
deterministic output for a fixed seed, degradation-order correctness, and that
the input GT is never corrupted.

Run:  python tests/test_degradation.py   (or: python -m pytest tests/ -q)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.degradation import (  # noqa: E402
    DegradationConfig,
    DegradationSimulator,
    make_rng,
)


def _gt(mean=0.5, size=256, seed=0):
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(mean, 0.15, (size, size)), 0, 1).astype(np.float32)
    return a


def test_output_dimensions():
    sim = DegradationSimulator(DegradationConfig(scale=2))
    out = sim.degrade(_gt(size=256), make_rng(1))
    assert out.shape == (128, 128)
    # scale 4 -> 64
    sim4 = DegradationSimulator(DegradationConfig(scale=4))
    assert sim4.degrade(_gt(size=256), make_rng(1)).shape == (64, 64)


def test_value_handling_dtype_and_range():
    sim = DegradationSimulator(DegradationConfig())  # clip_output False by default
    bright = np.full((256, 256), 0.95, np.float32)
    out = sim.degrade(bright, make_rng(3))
    assert out.dtype == np.float32
    # speckle on a bright image must overshoot > 1 when not clipped (like real NoisyLR)
    assert out.max() > 1.0

    sim_clip = DegradationSimulator(DegradationConfig(clip_output=True))
    out_c = sim_clip.degrade(bright, make_rng(3))
    assert out_c.min() >= 0.0 and out_c.max() <= 1.0


def test_deterministic_with_fixed_seed():
    sim = DegradationSimulator(DegradationConfig(severity_range=(0.7, 1.3)))
    gt = _gt()
    a = sim.degrade(gt, make_rng(42, 7))
    b = sim.degrade(gt, make_rng(42, 7))
    c = sim.degrade(gt, make_rng(43, 7))
    assert np.array_equal(a, b), "same seed must give identical output"
    assert not np.array_equal(a, c), "different seed must change output"


def test_degradation_order_is_respected():
    """Speckle applied at LR (downsample first) must have larger LR noise std than
    speckle applied at HR then downsampled (averaging reduces variance)."""
    gt = np.full((256, 256), 0.5, np.float32)  # constant -> isolates noise
    lr_noise = DegradationSimulator(DegradationConfig(order=("downsample", "speckle"), gauss_sigma=0.0))
    hr_noise = DegradationSimulator(DegradationConfig(order=("speckle", "downsample"), gauss_sigma=0.0))
    s_lr = lr_noise.degrade(gt, make_rng(5)).std()
    s_hr = hr_noise.degrade(gt, make_rng(5)).std()
    assert s_lr > s_hr, f"order not respected: LR-noise std {s_lr:.4f} !> HR-then-DS std {s_hr:.4f}"


def test_gt_is_not_corrupted():
    sim = DegradationSimulator(DegradationConfig(severity_range=(0.5, 1.5)))
    gt = _gt(seed=11)
    pristine = gt.copy()
    for i in range(5):
        _ = sim.degrade(gt, make_rng(1, i))
    assert np.array_equal(gt, pristine), "degrade() must not modify the input GT"


def test_config_roundtrip():
    cfg = DegradationConfig.from_dict({"order": ["speckle", "gaussian", "downsample"],
                                       "speckle_sigma": 0.2, "severity_range": [0.8, 1.2]})
    d = cfg.to_dict()
    assert tuple(d["order"]) == ("speckle", "gaussian", "downsample")
    assert d["speckle_sigma"] == 0.2


if __name__ == "__main__":
    test_output_dimensions()
    test_value_handling_dtype_and_range()
    test_deterministic_with_fixed_seed()
    test_degradation_order_is_respected()
    test_gt_is_not_corrupted()
    test_config_roundtrip()
    print("all degradation tests passed")
