"""KLA competition inference — standalone, evaluator-facing.

Loads a trained checkpoint, restores every degraded .npy in --input_dir, and
writes each restored image to --output_dir under the SAME filename. Supports
NVIDIA GPU with batching. No source edits, notebook cells, or hardcoded paths
are required to run it.

Contract (matches the official dataset):
    input : <id>.npy   float32  (128, 128)   NoisyLR, may exceed [0, 1]
    output: <id>.npy   float32  (256, 256)   restored, clipped to [0, 1]

Usage:
    python inference.py --input_dir ./input --output_dir ./output
    python inference.py --input_dir ./input --output_dir ./output \
        --checkpoint weights/final.pt --batch 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.inference.restorer import Restorer, load_restorer
from src.utils.npy_io import load_npy, save_restored


def parse_args():
    ap = argparse.ArgumentParser(description="KLA image restoration inference")
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("weights/final.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    ap.add_argument("--no-fp16", action="store_true", help="disable mixed precision")
    return ap.parse_args()


def resolve_device(requested: str | None) -> str:
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy files found in {args.input_dir}")

    model, meta = load_restorer(args.checkpoint, device=device)
    restorer = Restorer(model, device=device, fp16=not args.no_fp16)
    print(f"device={device}  checkpoint={args.checkpoint}  images={len(files)}  batch={args.batch}")

    t0 = time.perf_counter()
    n = 0
    for start in range(0, len(files), args.batch):
        chunk = files[start:start + args.batch]
        # Stack same-shaped inputs; fall back to per-image if shapes differ.
        arrs = [load_npy(p) for p in chunk]
        if len({a.shape for a in arrs}) == 1:
            preds = restorer.restore_batch(np.stack(arrs).astype(np.float32))
            for p, pred in zip(chunk, preds):
                save_restored(args.output_dir / p.name, pred)
                n += 1
        else:
            for p, a in zip(chunk, arrs):
                pred = restorer.restore_batch(a[None].astype(np.float32))[0]
                save_restored(args.output_dir / p.name, pred)
                n += 1
    dt = time.perf_counter() - t0
    print(f"restored {n} images in {dt:.2f}s  ({dt / n * 1000:.1f} ms/image, "
          f"{n / dt:.1f} img/s end-to-end)")


if __name__ == "__main__":
    main()
