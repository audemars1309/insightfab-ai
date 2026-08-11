"""Unified model registry.

One place that maps a model name + kwargs to an nn.Module, used by training,
the shared Restorer, and inference. Adding a new architecture means adding one
entry here; nothing else needs to know the class.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.baseline_cnn import SmallSRCNN
from src.models.nafnet_sr import NAFNetSR

_REGISTRY = {
    "small_srcnn": SmallSRCNN,
    "nafnet_sr": NAFNetSR,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model '{name}'. known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
