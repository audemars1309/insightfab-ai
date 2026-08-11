"""Data-integrity validation for the KLA semiconductor restoration dataset.

Runs a hard pass/fail check over every file so we never train on silently
corrupted or mismatched data. Verifies:

* every .npy opens and is finite (no NaN/Inf)
* GT shape (256, 256) float32 and strictly within [0, 1]
* NoisyLR shape (128, 128) float32
* GT <-> NoisyLR filename pairing is exact (no orphans)
* quantifies how far NoisyLR extends outside [0, 1] (expected, not an error)

Usage:
    python scripts/validate_data.py
Exit code 0 = all checks passed; 1 = at least one hard failure.
"""

from __future__ import annotations

import sys

import numpy as np

# Allow running as a plain script (python scripts/validate_data.py).
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.utils import paths  # noqa: E402
from src.utils.npy_io import GT_SHAPE, NOISY_SHAPE, load_npy  # noqa: E402


def _check_dir(directory, expected_shape, label, require_unit_range):
    """Return (n, failures, out_of_range_fraction_stats)."""
    failures: list[str] = []
    below0 = 0
    above1 = 0
    total_px = 0
    files = sorted(directory.glob("*.npy"))
    for p in files:
        try:
            arr = load_npy(p)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the audit
            failures.append(f"{label}/{p.name}: cannot load ({exc})")
            continue
        if arr.shape != expected_shape:
            failures.append(f"{label}/{p.name}: shape {arr.shape} != {expected_shape}")
        if arr.dtype != np.float32:
            failures.append(f"{label}/{p.name}: dtype {arr.dtype} != float32")
        if not np.isfinite(arr).all():
            failures.append(f"{label}/{p.name}: contains NaN/Inf")
        if require_unit_range:
            amin, amax = float(arr.min()), float(arr.max())
            # tiny float tolerance around the [0,1] normalization
            if amin < -1e-4 or amax > 1.0 + 1e-4:
                failures.append(f"{label}/{p.name}: GT out of [0,1] ({amin:.4f}..{amax:.4f})")
        below0 += int((arr < 0.0).sum())
        above1 += int((arr > 1.0).sum())
        total_px += arr.size
    stats = {
        "pct_below0": 100.0 * below0 / total_px if total_px else 0.0,
        "pct_above1": 100.0 * above1 / total_px if total_px else 0.0,
    }
    return len(files), failures, stats


def main() -> int:
    all_failures: list[str] = []

    n_gt, f_gt, s_gt = _check_dir(paths.TRAIN_GT_DIR, GT_SHAPE, "GT", require_unit_range=True)
    n_no, f_no, s_no = _check_dir(paths.TRAIN_NOISY_DIR, NOISY_SHAPE, "NoisyLR", require_unit_range=False)
    n_te, f_te, s_te = _check_dir(paths.TEST_NOISY_DIR, NOISY_SHAPE, "TestNoisyLR", require_unit_range=False)
    all_failures += f_gt + f_no + f_te

    # Pairing check
    gt_ids = set(paths.stem_ids(paths.TRAIN_GT_DIR))
    no_ids = set(paths.stem_ids(paths.TRAIN_NOISY_DIR))
    gt_only = sorted(gt_ids - no_ids)
    no_only = sorted(no_ids - gt_ids)
    if gt_only:
        all_failures.append(f"GT without NoisyLR pair: {len(gt_only)} (e.g. {gt_only[:5]})")
    if no_only:
        all_failures.append(f"NoisyLR without GT pair: {len(no_only)} (e.g. {no_only[:5]})")

    print("=" * 60)
    print("KLA dataset integrity validation")
    print("=" * 60)
    print(f"train GT       : {n_gt} files  (out-of-range as expected: "
          f"<0 {s_gt['pct_below0']:.4f}% / >1 {s_gt['pct_above1']:.4f}%)")
    print(f"train NoisyLR  : {n_no} files  (<0 {s_no['pct_below0']:.4f}% / >1 {s_no['pct_above1']:.4f}%)")
    print(f"test  NoisyLR  : {n_te} files  (<0 {s_te['pct_below0']:.4f}% / >1 {s_te['pct_above1']:.4f}%)")
    print(f"pairing        : {len(gt_ids & no_ids)} matched, "
          f"{len(gt_only)} GT-only, {len(no_only)} NoisyLR-only")
    print("-" * 60)
    if all_failures:
        print(f"FAILED: {len(all_failures)} problem(s)")
        for msg in all_failures[:50]:
            print("  -", msg)
        if len(all_failures) > 50:
            print(f"  ... and {len(all_failures) - 50} more")
        return 1
    print("PASSED: all files valid, shapes/dtypes correct, pairing exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
