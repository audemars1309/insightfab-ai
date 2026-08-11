"""KLA restoration — reproducible training entrypoint.

Config-driven so the exact submitted checkpoint can be reproduced from a YAML
plus a fixed seed. CLI flags override the most common config fields for quick
sweeps without editing files.

Examples:
    # Full NAFNet-SR training on a 16 GB GPU (Kaggle/Colab):
    python train.py --config configs/nafnet_sr_16gb.yaml

    # Quick pipeline check:
    python train.py --config configs/nafnet_sr_16gb.yaml --epochs 2 --limit-train 128

    # Reproduce the small baseline through the same engine:
    python train.py --config configs/small_srcnn.yaml

    # Resume an interrupted run:
    python train.py --config configs/nafnet_sr_16gb.yaml --resume weights/<exp_id>/resume.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.training.trainer import Trainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None)
    # common overrides (None => use config value)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lr-patch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", type=str)
    ap.add_argument("--exp-id", type=str)
    ap.add_argument("--limit-train", type=int)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    return ap.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    if args.epochs is not None:      cfg["train"]["epochs"] = args.epochs
    if args.batch is not None:       cfg["train"]["batch"] = args.batch
    if args.lr_patch is not None:    cfg["train"]["lr_patch"] = args.lr_patch
    if args.workers is not None:     cfg["train"]["workers"] = args.workers
    if args.limit_train is not None: cfg["train"]["limit_train"] = args.limit_train
    if args.lr is not None:          cfg["optim"]["lr"] = args.lr
    if args.seed is not None:        cfg["seed"] = args.seed
    if args.device is not None:      cfg["device"] = args.device
    if args.exp_id is not None:      cfg["exp_id"] = args.exp_id
    if args.no_amp:                  cfg["amp"] = False
    if args.no_ema:                  cfg["use_ema"] = False
    return cfg


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg = apply_overrides(cfg, args)
    cfg.setdefault("seed", 42)

    trainer = Trainer(cfg)
    trainer.maybe_resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
