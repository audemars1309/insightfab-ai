"""Weighted combination of restoration losses, built from config.

    L = w_pixel * Charbonnier
      + w_ssim   * (1 - SSIM)          # single-scale (baseline)
      + w_msssim * (1 - MS-SSIM)       # multi-scale structure (candidate)
      + w_hf     * HF-consistency      # anti-hallucination (candidate)
      + w_perceptual * VGG             # off by default

Backward compatible: w_msssim and w_hf default to 0.0, so existing configs
(e.g. the locked baseline's Charbonnier + 0.1*SSIM) produce identical losses.
Weights are tuned experimentally via config — see configs/*.yaml.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.charbonnier import CharbonnierLoss
from src.losses.ssim import SSIMLoss


class CombinedLoss(nn.Module):
    def __init__(self, w_pixel=1.0, w_ssim=0.0, w_msssim=0.0, w_hf=0.0,
                 w_perceptual=0.0, charbonnier_eps=1e-3):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_msssim = w_msssim
        self.w_hf = w_hf
        self.w_perceptual = w_perceptual
        self.pixel = CharbonnierLoss(eps=charbonnier_eps) if w_pixel > 0 else None
        self.ssim = SSIMLoss() if w_ssim > 0 else None
        self.msssim = None
        if w_msssim > 0:
            from src.losses.msssim import MSSSIMLoss
            self.msssim = MSSSIMLoss()
        self.hf = None
        if w_hf > 0:
            from src.losses.highfreq import HFConsistencyLoss
            self.hf = HFConsistencyLoss()
        self.perceptual = None
        if w_perceptual > 0:
            from src.losses.perceptual import VGGPerceptualLoss
            self.perceptual = VGGPerceptualLoss()

    def forward(self, pred, target):
        total = pred.new_zeros(())
        parts = {}
        if self.pixel is not None:
            p = self.pixel(pred, target)
            total = total + self.w_pixel * p
            parts["pixel"] = float(p.detach())
        if self.ssim is not None:
            s = self.ssim(pred, target)
            total = total + self.w_ssim * s
            parts["ssim"] = float(s.detach())
        if self.msssim is not None:
            m = self.msssim(pred, target)
            total = total + self.w_msssim * m
            parts["msssim"] = float(m.detach())
        if self.hf is not None:
            h = self.hf(pred, target)
            total = total + self.w_hf * h
            parts["hf"] = float(h.detach())
        if self.perceptual is not None:
            v = self.perceptual(pred, target)
            total = total + self.w_perceptual * v
            parts["perceptual"] = float(v.detach())
        return total, parts


def build_loss(cfg: dict) -> CombinedLoss:
    cfg = cfg or {}
    return CombinedLoss(
        w_pixel=cfg.get("w_pixel", 1.0),
        w_ssim=cfg.get("w_ssim", 0.0),
        w_msssim=cfg.get("w_msssim", 0.0),
        w_hf=cfg.get("w_hf", 0.0),
        w_perceptual=cfg.get("w_perceptual", 0.0),
        charbonnier_eps=cfg.get("charbonnier_eps", 1e-3),
    )
