"""Shared restoration engine used by both inference.py and benchmark.py.

Keeps a single, self-describing checkpoint format so inference never needs to
know architecture details at the call site:

    checkpoint = {
        "model_name": "small_srcnn",
        "model_kwargs": {"channels": 48, "n_blocks": 8},
        "state_dict": ...,
        "meta": {...},   # exp id, metrics, etc. (informational)
    }

Range policy matches the dataset rules: NoisyLR fed raw (not clipped); output
clipped to [0, 1] at save time (handled by src.utils.npy_io.save_restored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.registry import build_model


def save_checkpoint(path, model, model_name: str, model_kwargs: dict, meta: dict | None = None):
    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "meta": meta or {},
        },
        Path(path),
    )


def load_restorer(checkpoint_path, device: str = "cuda"):
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    model = build_model(ckpt["model_name"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("meta", {})


class Restorer:
    """Batched restorer over a list of (name, lr_array) inputs."""

    def __init__(self, model, device: str = "cuda", fp16: bool = True):
        self.model = model
        self.device = device
        self.fp16 = fp16 and device == "cuda"

    @torch.inference_mode()
    def restore_batch(self, lr_batch: np.ndarray) -> np.ndarray:
        """lr_batch: (B, H, W) float32 raw -> (B, 2H, 2W) float32 restored."""
        t = torch.from_numpy(lr_batch[:, None]).to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=self.fp16):
            out = self.model(t)
        return out[:, 0].float().cpu().numpy()
