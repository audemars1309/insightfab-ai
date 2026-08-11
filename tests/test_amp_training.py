"""Regression test for the AMP NaN-gradient freeze.

Bug: computing the SSIM/Charbonnier loss inside autocast (fp16) produced NaN
gradients (a multiply overflows), so GradScaler skipped every optimizer step and
the model never updated — validation metrics were frozen for the whole run.

Fix: compute the loss in fp32 (pred.float()) outside autocast. These tests assert
that, under AMP, one training step yields finite gradients AND actually updates
the parameters. The NaN only manifests on CUDA fp16, so the AMP test is skipped
on machines without CUDA; a CPU-safe fp32 sanity check always runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses.combined import build_loss  # noqa: E402
from src.models.registry import build_model  # noqa: E402

LOSS_CFG = {"w_pixel": 1.0, "w_ssim": 0.10, "w_perceptual": 0.0, "charbonnier_eps": 1e-3}


def _small_model():
    return build_model("nafnet_sr", img_channels=1, width=16,
                       enc_blocks=[1, 1], middle_blocks=1, dec_blocks=[1, 1], scale=2)


def test_combined_loss_fp32_grad_is_finite():
    """CPU-safe: the combined loss has finite gradients in fp32."""
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    crit = build_loss(LOSS_CFG)
    loss, _ = crit(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all(), "combined-loss fp32 gradient is not finite"


def test_amp_training_step_updates_params():
    """Under AMP, one step must produce finite grads and actually move params
    (guards the fp16-loss NaN freeze)."""
    if not torch.cuda.is_available():
        print("CUDA not available; skipping AMP training-step test")
        return
    dev = "cuda"
    torch.manual_seed(0)
    model = _small_model().to(dev).train()
    crit = build_loss(LOSS_CFG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    lr_t = torch.rand(2, 1, 128, 128, device=dev) * 1.4 - 0.1  # mimic NoisyLR range
    hr_t = torch.rand(2, 1, 256, 256, device=dev)

    before = next(model.parameters()).detach().clone()
    scale_before = scaler.get_scale()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            pred = model(lr_t)
        loss, _ = crit(pred.float(), hr_t)          # fp32 loss (the fix)
        scaler.scale(loss).backward()
        # gradients must be finite (no NaN/Inf) — the core regression assertion
        for p in model.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), "NaN/Inf gradient under AMP"
        scaler.step(opt)
        scaler.update()

    after = next(model.parameters()).detach()
    assert not torch.equal(before, after), "parameters did not update under AMP (model frozen)"
    assert scaler.get_scale() >= scale_before / 4, "GradScaler kept skipping steps (scale collapsed)"


if __name__ == "__main__":
    test_combined_loss_fp32_grad_is_finite()
    test_amp_training_step_updates_params()
    print("all AMP training tests passed")
