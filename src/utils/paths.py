"""Canonical project paths, resolved relative to the repo root.

Kept in one place so scripts never hardcode absolute local paths (a KLA
reproducibility requirement). Anything that must be evaluator-facing
(inference.py) takes directories as CLI arguments instead of using these.
"""

from __future__ import annotations

from pathlib import Path

# repo_root / src / utils / paths.py  ->  repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
TRAIN_GT_DIR = DATA_DIR / "train" / "GT"
TRAIN_NOISY_DIR = DATA_DIR / "train" / "NoisyLR"
TEST_NOISY_DIR = DATA_DIR / "test" / "NoisyLR"

CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
WEIGHTS_DIR = REPO_ROOT / "weights"
DOCS_DIR = REPO_ROOT / "docs"

AUDIT_DIR = RESULTS_DIR / "dataset_audit"


def stem_ids(directory: Path) -> list[str]:
    """Sorted list of .npy filename stems (e.g. '000298') in a directory."""
    return sorted(p.stem for p in directory.glob("*.npy"))
