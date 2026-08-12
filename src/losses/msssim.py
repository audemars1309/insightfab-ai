"""Multi-Scale SSIM loss (1 - MS-SSIM).

Why MS-SSIM (measured justification): on this data 96% of the recoverable
residual energy (GT - bicubic) lives in the low+mid frequency bands, and GT
high-frequency energy is near zero. Single-scale SSIM under-weights the
low/mid-scale structure that actually matters here; MS-SSIM supervises structure
across exactly those scales.

Standard Wang et al. (2003) formulation: per-scale contrast-structure (cs) plus
the full SSIM at the coarsest scale, combined with the canonical 5-scale weights.
Computed in fp32 (the trainer passes pred.float()); inputs are single-channel in
[0,1] (data_range = 1.0). Returns 1 - MS-SSIM (0 = identical, larger = worse).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.ssim import _gaussian_window

# Canonical MS-SSIM scale weights (sum to 1).
_MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


class MSSSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0,
                 weights=_MS_WEIGHTS):
        super().__init__()
        w1d = _gaussian_window(window_size, sigma)
        window = (w1d[:, None] @ w1d[None, :])[None, None]
        self.register_buffer("window", window)
        self.window_size = window_size
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def _ssim_and_cs(self, x, y):
        w = self.window.to(x.dtype)
        pad = self.window_size // 2
        mu_x = F.conv2d(x, w, padding=pad)
        mu_y = F.conv2d(y, w, padding=pad)
        mx2, my2, mxy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x = F.conv2d(x * x, w, padding=pad) - mx2
        sig_y = F.conv2d(y * y, w, padding=pad) - my2
        sig_xy = F.conv2d(x * y, w, padding=pad) - mxy
        cs = (2 * sig_xy + self.C2) / (sig_x + sig_y + self.C2)
        ssim = ((2 * mxy + self.C1) / (mx2 + my2 + self.C1)) * cs
        return ssim.mean(), cs.mean()

    def _max_levels(self, hw: int) -> int:
        # keep the coarsest scale at least ~window_size wide
        lv = 1
        while hw // 2 >= self.window_size and lv < self.weights.numel():
            hw //= 2
            lv += 1
        return lv

    def forward(self, pred, target):
        pred = pred.clamp(0.0, 1.0)
        levels = self._max_levels(min(pred.shape[-2:]))
        w = self.weights[:levels].to(pred.device)
        w = w / w.sum()  # renormalize if fewer scales fit (small patches)

        x, y = pred, target
        mcs = []
        last_ssim = None
        for i in range(levels):
            ssim_i, cs_i = self._ssim_and_cs(x, y)
            if i < levels - 1:
                mcs.append(cs_i.clamp(min=1e-6))
                x = F.avg_pool2d(x, kernel_size=2)
                y = F.avg_pool2d(y, kernel_size=2)
            else:
                last_ssim = ssim_i.clamp(min=1e-6)

        msssim = last_ssim ** w[-1]
        for i in range(levels - 1):
            msssim = msssim * (mcs[i] ** w[i])
        return 1.0 - msssim
