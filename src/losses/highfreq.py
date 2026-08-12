"""High-frequency consistency loss (anti-hallucination).

Why (measured justification): GT high-frequency energy is near zero
(HF fraction ~0.003 on train), and the baseline's validation decays as it starts
adding spurious high-frequency content past its peak. This term pushes the
prediction's high-frequency band toward the GT's high-frequency band, penalizing
BOTH fabricated detail (where GT is smooth) and missing detail (at real edges).

High-pass = image - Gaussian-blur(image). Loss = L1( HP(pred), HP(target) ).
0 when pred and target share the same high-frequency content. Computed in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.ssim import _gaussian_window


class HFConsistencyLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 2.0):
        super().__init__()
        w1d = _gaussian_window(window_size, sigma)
        kernel = (w1d[:, None] @ w1d[None, :])[None, None]
        self.register_buffer("blur", kernel)
        self.pad = window_size // 2

    def _high_pass(self, x):
        low = F.conv2d(x, self.blur.to(x.dtype), padding=self.pad)
        return x - low

    def forward(self, pred, target):
        return F.l1_loss(self._high_pass(pred.clamp(0.0, 1.0)), self._high_pass(target))
