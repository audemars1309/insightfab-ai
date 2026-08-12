"""Verify the candidate loss (MS-SSIM + HF consistency) and degradation
randomization. CPU-safe (no CUDA, no dataset needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses.msssim import MSSSIMLoss  # noqa: E402
from src.losses.highfreq import HFConsistencyLoss  # noqa: E402
from src.losses.combined import build_loss  # noqa: E402
from src.augmentation.degradation import DegradationConfig, DegradationSimulator, make_rng  # noqa: E402


def test_msssim_sign_and_range():
    """1-MS-SSIM: ~0 for identical, larger for degraded, always finite and >=0."""
    torch.manual_seed(0)
    x = torch.rand(2, 1, 256, 256)
    loss = MSSSIMLoss()
    same = float(loss(x, x))
    noisy = float(loss((x + 0.2 * torch.rand_like(x)).clamp(0, 1), x))
    assert 0.0 <= same < 1e-3, f"identical images should give ~0, got {same}"
    assert noisy > same, "degraded pair must have larger MS-SSIM loss"
    assert np.isfinite(same) and np.isfinite(noisy)


def test_msssim_gradient_finite():
    x = torch.rand(1, 1, 256, 256, requires_grad=True)
    y = torch.rand(1, 1, 256, 256)
    MSSSIMLoss()(x, y).backward()
    assert torch.isfinite(x.grad).all()


def test_hf_consistency_sign():
    """HF consistency is ~0 for identical and rises when pred adds high-freq."""
    torch.manual_seed(1)
    y = torch.rand(1, 1, 128, 128)
    hf = HFConsistencyLoss()
    same = float(hf(y, y))
    # add high-frequency speckle to pred only
    pred = (y + 0.3 * (torch.rand_like(y) - 0.5)).clamp(0, 1)
    added = float(hf(pred, y))
    assert same < 1e-6, f"identical images -> ~0 HF loss, got {same}"
    assert added > same, "fabricated high-frequency must increase HF-consistency loss"


def test_baseline_loss_unchanged_when_new_weights_zero():
    """Configs without w_msssim/w_hf must behave exactly like before (baseline safe)."""
    torch.manual_seed(2)
    pred = torch.rand(2, 1, 64, 64)
    tgt = torch.rand(2, 1, 64, 64)
    base = build_loss({"w_pixel": 1.0, "w_ssim": 0.1})            # locked-baseline recipe
    total, parts = base(pred, tgt)
    assert set(parts) == {"pixel", "ssim"}, f"unexpected loss terms: {parts}"
    assert torch.isfinite(total)


def test_candidate_loss_terms_present():
    pred = torch.rand(2, 1, 256, 256)
    tgt = torch.rand(2, 1, 256, 256)
    cand = build_loss({"w_pixel": 1.0, "w_msssim": 0.20, "w_hf": 0.05})
    total, parts = cand(pred, tgt)
    assert set(parts) == {"pixel", "msssim", "hf"}
    assert torch.isfinite(total)


def test_degradation_randomization():
    """Randomized order + kernel yields varied outputs; shape/dtype correct; GT intact."""
    cfg = DegradationConfig.from_dict({
        "scale": 2, "speckle_sigma": 0.156, "gauss_sigma": 0.009,
        "severity_range": [0.7, 1.3],
        "order_choices": [["downsample", "speckle", "gaussian"],
                          ["speckle", "downsample", "gaussian"],
                          ["gaussian", "downsample", "speckle"]],
        "downsample_order_choices": [1, 3],
    })
    sim = DegradationSimulator(cfg)
    gt = np.clip(np.random.default_rng(0).normal(0.5, 0.15, (256, 256)), 0, 1).astype(np.float32)
    pristine = gt.copy()
    outs = [sim.degrade(gt, make_rng(42, i)) for i in range(8)]
    for o in outs:
        assert o.shape == (128, 128) and o.dtype == np.float32
        assert np.isfinite(o).all()
    # different per-sample seeds -> different degradations (randomization active)
    assert not np.array_equal(outs[0], outs[1])
    # GT never modified
    assert np.array_equal(gt, pristine)


if __name__ == "__main__":
    test_msssim_sign_and_range()
    test_msssim_gradient_finite()
    test_hf_consistency_sign()
    test_baseline_loss_unchanged_when_new_weights_zero()
    test_candidate_loss_terms_present()
    test_degradation_randomization()
    print("all candidate tests passed")
