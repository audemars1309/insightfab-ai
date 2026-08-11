"""Paired NoisyLR/GT dataset for the KLA restoration task.

* Training mode: random aligned LR/HR patch crops + intensity-preserving
  augmentation (flips, 90-degree rotations). Patch training gives many samples
  per image and fits small GPUs.
* Eval mode: full 128->256 images, no augmentation.

Range policy: NoisyLR is returned raw (never clipped) so the model sees the
real out-of-[0,1] signal; GT stays in [0,1]. Arrays are shaped (1, H, W).

Only geometric, intensity-preserving augmentations are used — nothing that
would alter the meaning of semiconductor structures (no photometric jitter,
no elastic warps).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import paths
from src.utils.npy_io import SCALE_FACTOR, load_npy


class PairDataset(Dataset):
    def __init__(self, ids: list[str], train: bool, lr_patch: int = 64, seed: int = 0):
        self.ids = list(ids)
        self.train = train
        self.lr_patch = lr_patch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def _augment(self, lr: np.ndarray, hr: np.ndarray):
        # random 8-fold dihedral (flips + rot90); geometric only.
        if self.rng.random() < 0.5:
            lr, hr = lr[:, ::-1], hr[:, ::-1]
        if self.rng.random() < 0.5:
            lr, hr = lr[::-1, :], hr[::-1, :]
        k = int(self.rng.integers(0, 4))
        if k:
            lr, hr = np.rot90(lr, k), np.rot90(hr, k)
        return np.ascontiguousarray(lr), np.ascontiguousarray(hr)

    def __getitem__(self, i):
        stem = self.ids[i]
        lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")   # 128x128, raw range
        hr = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")      # 256x256, [0,1]
        if self.train:
            p = self.lr_patch
            H, W = lr.shape
            y = int(self.rng.integers(0, H - p + 1))
            x = int(self.rng.integers(0, W - p + 1))
            lr = lr[y:y + p, x:x + p]
            hr = hr[y * SCALE_FACTOR:(y + p) * SCALE_FACTOR,
                    x * SCALE_FACTOR:(x + p) * SCALE_FACTOR]
            lr, hr = self._augment(lr, hr)
        lr_t = torch.from_numpy(lr.astype(np.float32))[None]
        hr_t = torch.from_numpy(hr.astype(np.float32))[None]
        return lr_t, hr_t, stem


def load_split(path: Path | None = None) -> dict:
    path = path or (paths.CONFIGS_DIR / "split.json")
    return json.loads(Path(path).read_text())
