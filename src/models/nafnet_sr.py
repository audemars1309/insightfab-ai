"""NAFNet-SR — NAFNet restoration backbone with a PixelShuffle x2 SR tail.

Why NAFNet
----------
NAFNet (Chen et al., ECCV 2022, "Simple Baselines for Image Restoration") is a
strong, *activation-free* restoration backbone with excellent quality-per-FLOP,
which directly serves the KLA end-to-end-throughput axis. Its building block
uses LayerNorm, depthwise conv, a SimpleGate (multiplicative gating instead of
GELU/ReLU) and Simplified Channel Attention.

Adapting it to this task (joint 2x SR + denoise)
------------------------------------------------
* The UNet body runs entirely at LR resolution (denoising where the noise
  lives) and returns to the input resolution.
* A PixelShuffle x2 tail then super-resolves LR->HR without checkerboard
  artifacts.
* A global bicubic-upsample skip means the network learns the *residual* over
  bicubic, which stabilizes early training (same idea as the SmallSRCNN
  baseline, so comparisons are apples-to-apples).

All sizes are config-driven so one architecture spans a light 16 GB-friendly
setting and heavier variants.

Reference: https://github.com/megvii-research/NAFNet  (MIT licence)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over (N, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)  # depthwise
        self.conv3 = nn.Conv2d(dw // 2, c, 1)                    # after SimpleGate halves ch
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.sg = SimpleGate()

        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.drop2(x) * self.gamma


class NAFNetSR(nn.Module):
    def __init__(
        self,
        img_channels: int = 1,
        width: int = 32,
        enc_blocks=(2, 2, 4, 8),
        middle_blocks: int = 12,
        dec_blocks=(2, 2, 2, 2),
        scale: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        self.scale = scale
        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(middle_blocks)])

        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(n)]))

        # SR tail: LR features (width ch) -> HR (img_channels), no checkerboard.
        self.sr_tail = nn.Sequential(
            nn.Conv2d(width, width * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, img_channels, 3, padding=1),
        )
        self.padder_size = 2 ** len(enc_blocks)

    def _check_pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, h, w

    def forward(self, inp):
        base = F.interpolate(inp, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x, H, W = self._check_pad(inp)

        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        out = self.sr_tail(feat)                       # HR residual, (N,1,scale*Hpad,scale*Wpad)
        out = out[:, :, : H * self.scale, : W * self.scale]
        return base + out
