"""Baseline 0: bicubic 2x upsampling. The no-learning floor.

Upsamples each NoisyLR (128x128) to 256x256 with bicubic interpolation, clips
to [0,1], and scores PSNR/SSIM/LPIPS on the IID and OOD validation splits.
Establishes the minimum benchmark every learned model must beat.

Usage:
    python scripts/eval_bicubic.py
Writes experiments/baseline_bicubic.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from skimage.transform import resize as sk_resize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import paths  # noqa: E402
from src.utils.npy_io import GT_SHAPE, load_npy  # noqa: E402
from src.evaluation import metrics  # noqa: E402


def bicubic_up(noisy: np.ndarray) -> np.ndarray:
    return sk_resize(noisy, GT_SHAPE, order=3, preserve_range=True).astype(np.float32)


def eval_split(ids: list[str], device: str) -> dict:
    recs = []
    t0 = time.perf_counter()
    for stem in ids:
        noisy = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
        gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        pred = bicubic_up(noisy)
        recs.append(metrics.evaluate_pair(pred, gt, use_lpips=True, device=device))
    dt = time.perf_counter() - t0
    agg = metrics.aggregate(recs)
    agg["n"] = len(ids)
    agg["sec_per_image"] = dt / max(1, len(ids))
    return agg


def main():
    split = json.loads((paths.CONFIGS_DIR / "split.json").read_text())
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"LPIPS device: {device}")

    result = {
        "baseline": "bicubic_x2",
        "val_iid": eval_split(split["val_iid"], device),
        "val_ood": eval_split(split["val_ood"], device),
    }
    paths.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.EXPERIMENTS_DIR / "baseline_bicubic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
