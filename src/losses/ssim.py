"""Differentiable single-scale SSIM loss (1 - SSIM).

SSIM is one of the official metrics, so optimizing a structural term directly
tends to raise validation SSIM. Implemented with a Gaussian window; inputs are
single-channel in [0, 1] (data_range = 1.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        w1d = _gaussian_window(window_size, sigma)
        window = (w1d[:, None] @ w1d[None, :])[None, None]  # (1,1,ws,ws)
        self.register_buffer("window", window)
        self.window_size = window_size
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def _ssim_map(self, x, y):
        w = self.window.to(x.dtype)
        pad = self.window_size // 2
        mu_x = F.conv2d(x, w, padding=pad)
        mu_y = F.conv2d(y, w, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x = F.conv2d(x * x, w, padding=pad) - mu_x2
        sig_y = F.conv2d(y * y, w, padding=pad) - mu_y2
        sig_xy = F.conv2d(x * y, w, padding=pad) - mu_xy
        num = (2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sig_x + sig_y + self.C2)
        return num / (den + 1e-12)

    def forward(self, pred, target):
        # SSIM is defined on valid images; clamp pred so the loss stays well-posed.
        pred = pred.clamp(0.0, 1.0)
        return 1.0 - self._ssim_map(pred, target).mean()
