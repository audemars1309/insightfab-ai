"""Minimal structured experiment tracking.

Each run writes one JSON record under experiments/ with a unique id so previous
runs are never overwritten (a KLA reproducibility requirement). A CSV index is
appended for quick comparison across runs.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.utils import paths

_CSV_COLUMNS = [
    "exp_id", "timestamp", "model", "params", "seed", "epochs", "batch",
    "lr_patch", "loss", "lr", "device",
    "iid_psnr", "iid_ssim", "iid_lpips", "ood_psnr", "ood_ssim", "ood_lpips",
    "train_sec", "checkpoint",
]


def new_exp_id(prefix: str = "exp") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def save_record(record: dict) -> Path:
    """Write full JSON record + append a flat row to experiments/index.csv."""
    paths.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    record.setdefault("host", platform.node())
    record.setdefault("gpu", _gpu_name())
    exp_id = record["exp_id"]
    json_path = paths.EXPERIMENTS_DIR / f"{exp_id}.json"
    json_path.write_text(json.dumps(record, indent=2))

    index = paths.EXPERIMENTS_DIR / "index.csv"
    new_file = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        flat = {**record}
        for split in ("val_iid", "val_ood"):
            for m in ("psnr", "ssim", "lpips"):
                flat[f"{'iid' if split == 'val_iid' else 'ood'}_{m}"] = record.get(split, {}).get(m)
        w.writerow(flat)
    return json_path
