"""Optional VGG perceptual loss.

Off by default. Perceptual losses can improve LPIPS but also risk inventing
plausible-looking structure that isn't real — unacceptable for semiconductor
inspection. Enable only as a small-weight term and verify it does not degrade
PSNR/SSIM or hallucinate detail (inspect images, not just metrics).

Grayscale is replicated to 3 channels; features come from ImageNet-pretrained
VGG16 (torchvision, BSD-3). Disclosed in docs/external_resources.md when used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 15), resize: bool = False):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.layers = set(layers)
        self.resize = resize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _prep(self, x):
        x = x.clamp(0, 1).repeat(1, 3, 1, 1)  # gray -> 3ch
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p, t = self._prep(pred), self._prep(target)
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            p, t = layer(p), layer(t)
            if i in self.layers:
                loss = loss + nn.functional.l1_loss(p, t)
            if i >= max(self.layers):
                break
        return loss
