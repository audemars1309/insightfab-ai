"""Evaluation metrics for the KLA restoration task: PSNR, SSIM, LPIPS.

All three are the official reported metrics. PSNR/SSIM come from scikit-image
(no torch needed). LPIPS uses the reference `lpips` package on GPU/CPU; it is
lazily constructed so importing this module is cheap and torch-free until LPIPS
is actually requested.

Convention: inputs are single-channel float32 arrays in [0, 1], shape (H, W).
The restored output is clipped to [0, 1] before scoring, matching how we save.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# GT is in [0, 1]; PSNR needs the data range.
_DATA_RANGE = 1.0

_lpips_model = None  # lazy singleton


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    return float(peak_signal_noise_ratio(gt.astype(np.float32), pred, data_range=_DATA_RANGE))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    return float(structural_similarity(gt.astype(np.float32), pred, data_range=_DATA_RANGE))


def _get_lpips(device: str = "cpu"):
    global _lpips_model
    if _lpips_model is None:
        import lpips  # noqa: PLC0415
        import torch  # noqa: PLC0415

        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        _lpips_model._kla_device = device  # remember placement
    return _lpips_model


def lpips_score(pred: np.ndarray, gt: np.ndarray, device: str = "cpu") -> float:
    """LPIPS (AlexNet). Grayscale is replicated to 3 channels; inputs mapped to
    [-1, 1] as the LPIPS package expects."""
    import torch  # noqa: PLC0415

    model = _get_lpips(device)
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt = gt.astype(np.float32)

    def _to_tensor(x):
        t = torch.from_numpy(x)[None, None]        # (1,1,H,W)
        t = t.repeat(1, 3, 1, 1) * 2.0 - 1.0        # gray->3ch, [0,1]->[-1,1]
        return t.to(device)

    with torch.no_grad():
        d = model(_to_tensor(pred), _to_tensor(gt))
    return float(d.item())


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, use_lpips: bool = True,
                  device: str = "cpu") -> dict:
    out = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
    if use_lpips:
        out["lpips"] = lpips_score(pred, gt, device=device)
    return out


def aggregate(records: list[dict]) -> dict:
    """Mean of each metric over a list of per-image metric dicts."""
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}
