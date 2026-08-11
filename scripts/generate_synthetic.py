"""Offline generation of synthetic degraded/GT pairs from the official GT images.

Reproducible (seed-driven). Writes synthetic NoisyLR arrays for inspection or for
an offline synthetic dataset. Does NOT touch the official data or the experiment
index. Intended for visual sanity-checking and calibration inspection.

Usage:
    # generate 200 synthetic NoisyLR from the training GT ids
    python scripts/generate_synthetic.py --n 200 --out results/synthetic_preview \
        --config configs/degradation_default.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.degradation import DegradationConfig, DegradationSimulator, make_rng  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.npy_io import load_npy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "synthetic_preview")
    ap.add_argument("--config", type=Path, default=None, help="degradation yaml; default = measured defaults")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-arrays", action="store_true", help="also write the .npy arrays")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(args.config.read_text()) if args.config else {}
    cfg = DegradationConfig.from_dict(cfg_dict.get("degradation", cfg_dict))
    sim = DegradationSimulator(cfg)

    ids = paths.stem_ids(paths.TRAIN_GT_DIR)[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    stats = []
    for i, stem in enumerate(ids):
        gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        syn = sim.degrade(gt, make_rng(args.seed, i))
        stats.append({
            "id": stem, "shape": list(syn.shape),
            "min": float(syn.min()), "max": float(syn.max()),
            "mean": float(syn.mean()), "std": float(syn.std()),
            "pct_above1": float((syn > 1).mean() * 100),
            "pct_below0": float((syn < 0).mean() * 100),
        })
        if args.save_arrays:
            np.save(args.out / f"{stem}.npy", syn)

    summary = {
        "config": cfg.to_dict(),
        "n": len(ids),
        "mean_pct_above1": float(np.mean([s["pct_above1"] for s in stats])),
        "mean_pct_below0": float(np.mean([s["pct_below0"] for s in stats])),
        "mean_std": float(np.mean([s["std"] for s in stats])),
    }
    (args.out / "synthetic_summary.json").write_text(json.dumps({"summary": summary, "per_image": stats[:20]}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'synthetic_summary.json'}"
          + (f" and {len(ids)} arrays" if args.save_arrays else ""))


if __name__ == "__main__":
    main()
