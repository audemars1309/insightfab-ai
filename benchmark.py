"""End-to-end inference benchmark for the KLA restoration pipeline.

KLA scores *total* pipeline time, not just the forward pass. This measures each
stage separately so we can see where time goes:

    disk read -> preprocess/stack -> H2D transfer -> model -> D2H transfer
    -> postprocess/clip -> disk write

Reports model-only vs end-to-end runtime, throughput, batch size, device,
peak GPU memory, and library versions, with a warmup pass excluded from timing.

Usage:
    python benchmark.py --input_dir data/test/NoisyLR --checkpoint weights/final.pt --batch 16
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.inference.restorer import load_restorer
from src.utils.npy_io import load_npy, save_restored


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=Path("data/test/NoisyLR"))
    ap.add_argument("--checkpoint", type=Path, default=Path("weights/final.pt"))
    ap.add_argument("--output_dir", type=Path, default=Path("results/benchmark_out"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = all files")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = (not args.no_fp16) and device == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npy"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npy in {args.input_dir}")

    model, _ = load_restorer(args.checkpoint, device=device)

    def run_batches(measure: bool):
        stage = {k: 0.0 for k in ["read", "pre", "h2d", "model", "d2h", "post", "write"]}
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(files), args.batch):
            chunk = files[start:start + args.batch]

            t = time.perf_counter()
            arrs = [load_npy(p) for p in chunk]
            stage["read"] += time.perf_counter() - t

            t = time.perf_counter()
            batch = np.stack(arrs).astype(np.float32)[:, None]
            cpu_t = torch.from_numpy(batch).pin_memory() if device == "cuda" else torch.from_numpy(batch)
            stage["pre"] += time.perf_counter() - t

            t = time.perf_counter()
            gpu_t = cpu_t.to(device, non_blocking=True)
            if device == "cuda":
                torch.cuda.synchronize()
            stage["h2d"] += time.perf_counter() - t

            t = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=fp16):
                out = model(gpu_t)
            if device == "cuda":
                torch.cuda.synchronize()
            stage["model"] += time.perf_counter() - t

            t = time.perf_counter()
            out_cpu = out.float().cpu().numpy()
            stage["d2h"] += time.perf_counter() - t

            t = time.perf_counter()
            out_cpu = np.clip(out_cpu[:, 0], 0.0, 1.0)
            stage["post"] += time.perf_counter() - t

            t = time.perf_counter()
            for p, arr in zip(chunk, out_cpu):
                save_restored(args.output_dir / p.name, arr)
            stage["write"] += time.perf_counter() - t
        return stage

    # Warmup (not timed) — triggers cudnn autotune, allocation, JIT.
    run_batches(measure=False)

    t0 = time.perf_counter()
    stage = run_batches(measure=True)
    total = time.perf_counter() - t0
    n = len(files)

    peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
    model_only = stage["model"]
    print("=" * 56)
    print("KLA end-to-end inference benchmark")
    print("=" * 56)
    print(f"device={device}  fp16={fp16}  batch={args.batch}  images={n}")
    print(f"GPU={torch.cuda.get_device_name(0) if device=='cuda' else '-'}")
    print(f"torch={torch.__version__}  python={platform.python_version()}")
    print("-" * 56)
    for k in ["read", "pre", "h2d", "model", "d2h", "post", "write"]:
        print(f"  {k:6s}: {stage[k]*1000:8.1f} ms total   {stage[k]/n*1000:7.2f} ms/img")
    print("-" * 56)
    print(f"model-only : {model_only:.3f} s   ({n/model_only:.1f} img/s)")
    print(f"END-TO-END : {total:.3f} s   ({n/total:.1f} img/s, {total/n*1000:.2f} ms/img)")
    print(f"peak GPU mem: {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
