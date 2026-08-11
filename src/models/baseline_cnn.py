"""SmallSRCNN — an efficient learned baseline for 2x restoration.

Design rationale
----------------
The task is joint denoise + 2x super-resolution. A minimal architecture that
does both:

  LR (1ch) --conv--> feature --[N residual blocks @ LR]--> --PixelShuffle x2-->
  HR feature --conv--> HR residual   (+)   bicubic-upsampled LR   ==> HR output

* Residual blocks at LR resolution keep compute cheap (denoising happens where
  the noise lives).
* PixelShuffle does the 2x upsample without checkerboard artifacts.
* A global skip that adds the bicubic upsample means the network only learns
  the *correction* to bicubic, which trains fast and stabilizes early epochs.

Intentionally small so it fits a 4 GB GPU and sets an honest learned floor;
it is a baseline, not the final competition model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x), inplace=True))


class SmallSRCNN(nn.Module):
    def __init__(self, channels: int = 48, n_blocks: int = 8, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        # Global skip: learn the residual over bicubic upsampling.
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        f = self.head(x)
        f = f + self.body(f)
        f = self.upsample(f)
        return base + self.tail(f)


def build_model(name: str = "small_srcnn", **kw) -> nn.Module:
    if name == "small_srcnn":
        return SmallSRCNN(**kw)
    raise ValueError(f"unknown model '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
