"""Weighted combination of restoration losses, built from config.

    L = w_pixel * Charbonnier + w_ssim * (1-SSIM) + w_perceptual * VGG

Defaults favor fidelity + structure (pixel + ssim) and keep perceptual off, so
the model does not hallucinate structure. Weights are meant to be tuned
experimentally via config — see configs/*.yaml.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.charbonnier import CharbonnierLoss
from src.losses.ssim import SSIMLoss


class CombinedLoss(nn.Module):
    def __init__(self, w_pixel=1.0, w_ssim=0.0, w_perceptual=0.0, charbonnier_eps=1e-3):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual
        self.pixel = CharbonnierLoss(eps=charbonnier_eps) if w_pixel > 0 else None
        self.ssim = SSIMLoss() if w_ssim > 0 else None
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
        w_perceptual=cfg.get("w_perceptual", 0.0),
        charbonnier_eps=cfg.get("charbonnier_eps", 1e-3),
    )
