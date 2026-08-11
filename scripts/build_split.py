"""Build a reproducible, leakage-free train/validation split.

Design (justified in docs/dataset_report.md):
* The atom is an image *pair* (one GT + its NoisyLR). We split at the pair
  level, so no patch cropped later can appear in both train and val.
* IID validation is stratified by content cluster so it mirrors the training
  distribution.
* OOD validation holds out one entire content cluster the model never trains
  on, approximating the hidden test set's "unfamiliar image content".

Output: configs/split.json  (fixed seed => byte-identical on re-run).

Usage:
    python scripts/build_split.py [--seed 42] [--iid-frac 0.10] [--ood-cluster 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iid-frac", type=float, default=0.10)
    ap.add_argument("--ood-cluster", type=int, default=6,
                    help="content cluster id held out entirely for OOD validation")
    args = ap.parse_args()

    clusters_path = paths.CONFIGS_DIR / "content_clusters.json"
    if not clusters_path.exists():
        raise SystemExit("configs/content_clusters.json missing; run scripts/audit_dataset.py first.")
    clusters: dict[str, int] = json.loads(clusters_path.read_text())

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for stem, lab in clusters.items():
        by_cluster[int(lab)].append(stem)
    for lab in by_cluster:
        by_cluster[lab].sort()

    rng = np.random.default_rng(args.seed)

    # OOD holdout: one full cluster.
    val_ood = sorted(by_cluster.get(args.ood_cluster, []))
    ood_set = set(val_ood)

    # Remaining pool, stratified IID validation sample per cluster.
    train, val_iid = [], []
    for lab, stems in sorted(by_cluster.items()):
        if lab == args.ood_cluster:
            continue
        stems = [s for s in stems if s not in ood_set]
        n_val = max(1, int(round(len(stems) * args.iid_frac)))
        idx = rng.permutation(len(stems))
        val_ids = {stems[i] for i in idx[:n_val]}
        for s in stems:
            (val_iid if s in val_ids else train).append(s)

    train.sort(); val_iid.sort()

    # Sanity: partitions are disjoint and cover the training-cluster pool.
    assert not (set(train) & set(val_iid)), "train/val_iid overlap"
    assert not (set(train) & ood_set), "train/ood overlap"
    assert not (set(val_iid) & ood_set), "val_iid/ood overlap"
    total = len(train) + len(val_iid) + len(val_ood)
    assert total == len(clusters), f"coverage mismatch {total} != {len(clusters)}"

    split = {
        "seed": args.seed,
        "iid_frac": args.iid_frac,
        "ood_cluster": args.ood_cluster,
        "counts": {"train": len(train), "val_iid": len(val_iid), "val_ood": len(val_ood)},
        "train": train,
        "val_iid": val_iid,
        "val_ood": val_ood,
    }
    out = paths.CONFIGS_DIR / "split.json"
    out.write_text(json.dumps(split, indent=2))
    print("wrote", out)
    print("counts:", split["counts"], "total", total)


if __name__ == "__main__":
    main()
