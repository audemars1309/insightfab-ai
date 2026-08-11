"""Charbonnier loss — a smooth, robust L1 variant.

Preferred over MSE for restoration: MSE over-penalizes outliers and tends to
oversmooth (which also hurts LPIPS). Charbonnier = sqrt((x-y)^2 + eps^2)
behaves like L1 away from zero but stays differentiable at zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))
