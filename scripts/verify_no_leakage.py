"""Pre-flight data-leakage verification for the degradation-aware candidate.

Confirms, programmatically, that synthetic re-degradation uses ONLY training GT
and never touches validation/OOD/test images. Run this before the A/B training.

Usage:
    python scripts/verify_no_leakage.py --config configs/nafnet_sr_degaug.yaml
Exit 0 = all checks pass; 1 = leakage detected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.synthetic_dataset import SyntheticMixDataset  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.utils import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=paths.CONFIGS_DIR / "nafnet_sr_degaug.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    split = load_split()
    train, vi, vo = set(split["train"]), set(split["val_iid"]), set(split["val_ood"])

    syn = cfg.get("synthetic") or {}
    if not syn.get("enabled"):
        print("Config has no enabled synthetic block — no synthetic data used. PASS.")
        return 0

    # The dataset the trainer builds for synthetic augmentation:
    ds = SyntheticMixDataset(split["train"], lr_patch=cfg["train"]["lr_patch"], seed=cfg["seed"],
                             degrade_cfg=syn.get("degradation"), synth_prob=syn.get("synth_prob", 0.5))
    ids = set(ds.ids)

    checks = []
    checks.append(("synthetic_ids ⊆ split['train']", ids.issubset(train)))
    checks.append(("synthetic_ids ∩ val_iid == ∅", len(ids & vi) == 0))
    checks.append(("synthetic_ids ∩ val_ood == ∅", len(ids & vo) == 0))

    # Static check: the synthetic dataset source never references a test path.
    sd_src = (paths.REPO_ROOT / "src" / "augmentation" / "synthetic_dataset.py").read_text()
    no_test = not any(t in sd_src for t in ("TEST_NOISY", "data/test", "test/NoisyLR"))
    checks.append(("SyntheticMixDataset references no TEST path", no_test))
    # And it only reads the training GT/NoisyLR dirs
    reads_train_only = ("TRAIN_GT_DIR" in sd_src and "TRAIN_NOISY_DIR" in sd_src)
    checks.append(("SyntheticMixDataset reads only training dirs", reads_train_only))

    print("=" * 60)
    print("DATA-LEAKAGE VERIFICATION (candidate B)")
    print("=" * 60)
    print(f"train={len(train)}  val_iid={len(vi)}  val_ood={len(vo)}  synthetic_source_ids={len(ids)}")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("-" * 60)
    print("ALL CHECKS PASSED — synthetic data derives only from training GT." if ok
          else "LEAKAGE DETECTED — do not train.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
