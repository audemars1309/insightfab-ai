"""Deliberate, documented .npy I/O for the KLA restoration task.

Why this module exists
----------------------
The official rules make normalization a *correctness* issue, not a convenience:

* GT is normalized to [0, 1].
* NoisyLR values may extend slightly outside [0, 1] on purpose (speckle
  overshoot above 1.0, additive-Gaussian tail below 0.0). We measured this
  directly on the provided data.
* KLA does NOT clip or renormalize our outputs. They score the array exactly
  as we save it, against the hidden [0, 1] GT.

Consequences enforced here:

1. On LOAD of a NoisyLR input we do NOT clip. Clipping the input would throw
   away real signal the model needs and bias bright regions downward.
2. On SAVE of a restored output we DO clip to [0, 1], because the target space
   is [0, 1] and any excursion outside it can only hurt the pixel metrics.
3. Everything stays float32 to match the dataset dtype exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Target value range of the clean GT / restored output.
GT_MIN = 0.0
GT_MAX = 1.0

# Expected shapes for the KLA benchmark (measured from the official dataset).
GT_SHAPE = (256, 256)
NOISY_SHAPE = (128, 128)
SCALE_FACTOR = 2  # 128 -> 256


def load_npy(path: str | Path) -> np.ndarray:
    """Load a .npy array as float32 without altering its values.

    We deliberately preserve out-of-range values here; range handling is the
    caller's decision and differs for inputs (keep) vs outputs (clip on save).
    """
    arr = np.load(Path(path))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def save_restored(path: str | Path, arr: np.ndarray) -> None:
    """Save a restored image as the competition expects it.

    Clips to [0, 1] and writes float32, because the scorer compares directly to
    [0, 1] GT and never renormalizes our file.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, GT_MIN, GT_MAX)
    np.save(Path(path), arr)


def to_chw(arr: np.ndarray) -> np.ndarray:
    """(H, W) grayscale -> (1, H, W) for a single-channel model input."""
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr
    raise ValueError(f"Expected (H, W) or (1, H, W) grayscale, got shape {arr.shape}")


def from_chw(arr: np.ndarray) -> np.ndarray:
    """(1, H, W) -> (H, W) for saving as the dataset's 2-D grayscale format."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected (1, H, W) or (H, W), got shape {arr.shape}")
