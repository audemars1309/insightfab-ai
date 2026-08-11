"""Synthetic-augmented training dataset for the Phase 9 ablation.

Standalone by design: it reuses shared utils (npy I/O, paths, the simulator) but
does NOT modify the validated `PairDataset`, so the default training path is
unaffected. Used only when a training config explicitly enables `synthetic`.

For each sampled GT image, with probability `synth_prob` the NoisyLR is generated
on the fly by the degradation simulator; otherwise the real official NoisyLR is
used. Cropping + geometric augmentation match `PairDataset` exactly so the two
sources are interchangeable.

Validation always uses the real official pairs — synthetic data augments training
only, never model selection.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation.degradation import DegradationConfig, DegradationSimulator, make_rng
from src.utils import paths
from src.utils.npy_io import SCALE_FACTOR, load_npy


class SyntheticMixDataset(Dataset):
    def __init__(self, ids, lr_patch: int = 128, seed: int = 42,
                 degrade_cfg: dict | None = None, synth_prob: float = 0.5):
        self.ids = list(ids)
        self.lr_patch = lr_patch
        self.seed = seed
        self.synth_prob = float(synth_prob)
        self.sim = DegradationSimulator(DegradationConfig.from_dict(degrade_cfg))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def _augment(self, lr, hr, rng):
        if rng.random() < 0.5:
            lr, hr = lr[:, ::-1], hr[:, ::-1]
        if rng.random() < 0.5:
            lr, hr = lr[::-1, :], hr[::-1, :]
        k = int(rng.integers(0, 4))
        if k:
            lr, hr = np.rot90(lr, k), np.rot90(hr, k)
        return np.ascontiguousarray(lr), np.ascontiguousarray(hr)

    def __getitem__(self, i):
        stem = self.ids[i]
        hr = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")           # 256x256, [0,1]
        # Per-item RNG makes synthetic generation reproducible per (seed, index).
        rng = make_rng(self.seed, i)
        if rng.random() < self.synth_prob:
            lr = self.sim.degrade(hr, rng)                          # synthetic NoisyLR
        else:
            lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")    # real NoisyLR

        p = self.lr_patch
        H, W = lr.shape
        y = int(rng.integers(0, H - p + 1))
        x = int(rng.integers(0, W - p + 1))
        lr = lr[y:y + p, x:x + p]
        hr = hr[y * SCALE_FACTOR:(y + p) * SCALE_FACTOR, x * SCALE_FACTOR:(x + p) * SCALE_FACTOR]
        lr, hr = self._augment(lr, hr, rng)

        lr_t = torch.from_numpy(lr.astype(np.float32))[None]
        hr_t = torch.from_numpy(hr.astype(np.float32))[None]
        return lr_t, hr_t, stem
