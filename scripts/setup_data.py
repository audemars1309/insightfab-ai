"""Prepare data/ from the official ZIPs or a Kaggle/Colab dataset mount.

Extracts train.zip + Test_NoisyLR.zip into the expected layout, excluding macOS
junk (__MACOSX/, .DS_Store). Idempotent: skips extraction if the target already
has the right file counts.

Usage:
    # From local ZIPs:
    python scripts/setup_data.py --train-zip semicon_dataset/train.zip \
        --test-zip semicon_dataset/Test_NoisyLR.zip

    # On Kaggle (dataset added at /kaggle/input/<slug>/):
    python scripts/setup_data.py --src /kaggle/input/semicon-dataset
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import paths  # noqa: E402


def _extract(zip_path: Path, dest: Path):
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if "__MACOSX" not in m and not m.endswith(".DS_Store")]
        z.extractall(dest, members=members)


def _count(p: Path) -> int:
    return len(list(p.glob("*.npy"))) if p.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-zip", type=Path)
    ap.add_argument("--test-zip", type=Path)
    ap.add_argument("--src", type=Path, help="dir containing train.zip + Test_NoisyLR.zip")
    args = ap.parse_args()

    train_zip = args.train_zip or (args.src / "train.zip" if args.src else None)
    test_zip = args.test_zip or (args.src / "Test_NoisyLR.zip" if args.src else None)
    if not train_zip or not test_zip:
        raise SystemExit("provide --train-zip/--test-zip or --src")

    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _count(paths.TRAIN_GT_DIR) != 3200:
        print(f"extracting {train_zip} ...")
        _extract(Path(train_zip), paths.DATA_DIR)          # -> data/train/{GT,NoisyLR}
    if _count(paths.TEST_NOISY_DIR) != 400:
        print(f"extracting {test_zip} ...")
        _extract(Path(test_zip), paths.DATA_DIR / "test")  # -> data/test/NoisyLR

    print(f"train GT      : {_count(paths.TRAIN_GT_DIR)}")
    print(f"train NoisyLR : {_count(paths.TRAIN_NOISY_DIR)}")
    print(f"test  NoisyLR : {_count(paths.TEST_NOISY_DIR)}")
    ok = (_count(paths.TRAIN_GT_DIR) == 3200 and _count(paths.TRAIN_NOISY_DIR) == 3200
          and _count(paths.TEST_NOISY_DIR) == 400)
    print("OK" if ok else "WARNING: unexpected counts")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
