"""Full-image validation matching the inference path (128 -> 256, no cropping).

PSNR/SSIM are computed on every validation image; LPIPS (slower) on the first
`lpips_n` images to keep per-epoch validation fast. Returns mean metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation import metrics
from src.utils import paths
from src.utils.npy_io import load_npy


@torch.no_grad()
def evaluate_model(model, ids, device: str, lpips_n: int = 64, fp16: bool = True) -> dict:
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for k, stem in enumerate(ids):
        lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
        gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        lr_t = torch.from_numpy(lr.astype(np.float32))[None, None].to(device)
        with torch.autocast(device_type="cuda", enabled=(fp16 and device == "cuda")):
            pred = model(lr_t)
        pred = pred[0, 0].float().cpu().numpy()
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        if k < lpips_n:
            lpipss.append(metrics.lpips_score(pred, gt, device=device))
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)) if lpipss else None,
        "n": len(ids),
        "lpips_n": min(lpips_n, len(ids)),
    }
