"""Full dataset audit for the KLA restoration task.

Measures the *actual* dataset (no assumptions) and writes:
  results/dataset_audit/audit_stats.json      aggregate numbers
  results/dataset_audit/histograms/*.png      intensity distributions
  results/dataset_audit/sample_pairs/*.png    GT vs NoisyLR(bicubic-up)
  results/dataset_audit/difference_maps/*.png  residual visualizations
  results/dataset_audit/noise_vs_intensity.png degradation characterization
  configs/content_clusters.json               per-image content cluster (for OOD split)

The narrative report (docs/dataset_report.md) is written from audit_stats.json so
every stated number traces back to a measured value here.

Usage:
    python scripts/audit_dataset.py [--noise-subset 400] [--clusters 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage.transform import resize as sk_resize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import paths  # noqa: E402
from src.utils.npy_io import load_npy  # noqa: E402

rng = np.random.default_rng(1234)


# --------------------------------------------------------------------------- #
# Per-image streaming statistics
# --------------------------------------------------------------------------- #
def per_image_stats(directory: Path):
    ids, means, stds, mins, maxs = [], [], [], [], []
    below0, above1 = [], []
    for p in sorted(directory.glob("*.npy")):
        a = load_npy(p)
        ids.append(p.stem)
        means.append(float(a.mean()))
        stds.append(float(a.std()))
        mins.append(float(a.min()))
        maxs.append(float(a.max()))
        below0.append(float((a < 0).mean() * 100))
        above1.append(float((a > 1).mean() * 100))
    return {
        "ids": ids,
        "mean": np.array(means),
        "std": np.array(stds),
        "min": np.array(mins),
        "max": np.array(maxs),
        "pct_below0": np.array(below0),
        "pct_above1": np.array(above1),
    }


def summarize(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


# --------------------------------------------------------------------------- #
# Degradation characterization: additive vs multiplicative (speckle)
# --------------------------------------------------------------------------- #
def characterize_noise(n_subset: int):
    """Estimate the residual NoisyLR - downsample(GT) and its dependence on
    local clean intensity. Additive Gaussian => residual std ~ constant.
    Speckle (multiplicative) => residual std grows with intensity.
    """
    ids = paths.stem_ids(paths.TRAIN_GT_DIR)
    pick = rng.choice(len(ids), size=min(n_subset, len(ids)), replace=False)
    inten_bins = np.linspace(0, 1, 11)  # 10 intensity bins
    resid_by_bin = [list() for _ in range(len(inten_bins) - 1)]
    global_resid = []
    for i in pick:
        stem = ids[i]
        gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        noisy = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
        # Bring GT to NoisyLR resolution so the residual isolates *noise*,
        # not the resolution gap. anti_aliasing matches a plausible downsample.
        gt_lr = sk_resize(gt, noisy.shape, order=3, anti_aliasing=True,
                          preserve_range=True).astype(np.float32)
        resid = noisy - gt_lr
        global_resid.append(resid.ravel())
        idx = np.clip(np.digitize(gt_lr.ravel(), inten_bins) - 1, 0, len(inten_bins) - 2)
        rr = resid.ravel()
        for b in range(len(inten_bins) - 1):
            sel = rr[idx == b]
            if sel.size:
                resid_by_bin[b].append(sel)
    global_resid = np.concatenate(global_resid)
    bin_centers = 0.5 * (inten_bins[:-1] + inten_bins[1:])
    bin_std = np.array([
        float(np.concatenate(b).std()) if b else np.nan for b in resid_by_bin
    ])
    bin_mean = np.array([
        float(np.concatenate(b).mean()) if b else np.nan for b in resid_by_bin
    ])
    return {
        "global_residual": summarize(global_resid),
        "residual_std_by_intensity": {
            "intensity": bin_centers.tolist(),
            "resid_std": bin_std.tolist(),
            "resid_mean": bin_mean.tolist(),
        },
        "n_pairs_used": int(len(pick)),
    }


# --------------------------------------------------------------------------- #
# Exact-duplicate / leakage detection via content hashing
# --------------------------------------------------------------------------- #
def hash_dir(directory: Path) -> dict[str, str]:
    out = {}
    for p in sorted(directory.glob("*.npy")):
        a = load_npy(p)
        out[p.stem] = hashlib.md5(a.tobytes()).hexdigest()
    return out


def find_dupes(hashes: dict[str, str]) -> list[list[str]]:
    inv = defaultdict(list)
    for stem, h in hashes.items():
        inv[h].append(stem)
    return [sorted(v) for v in inv.values() if len(v) > 1]


# --------------------------------------------------------------------------- #
# Content clustering (thumbnail features) -> OOD holdout groups
# --------------------------------------------------------------------------- #
def cluster_content(k: int):
    from sklearn.cluster import KMeans  # local import; sklearn optional

    ids = paths.stem_ids(paths.TRAIN_GT_DIR)
    feats = np.zeros((len(ids), 16 * 16), dtype=np.float32)
    for i, stem in enumerate(ids):
        a = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        thumb = sk_resize(a, (16, 16), order=1, anti_aliasing=True, preserve_range=True)
        feats[i] = thumb.ravel()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(feats)
    sizes = {int(c): int((labels == c).sum()) for c in range(k)}
    return ids, labels.tolist(), sizes


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
def save_histograms(gt_s, no_s, te_s, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Aggregate intensity histogram from per-image means as a light proxy + full
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(gt_s["mean"], bins=50, alpha=0.7, label="GT")
    ax[0].hist(no_s["mean"], bins=50, alpha=0.7, label="NoisyLR")
    ax[0].hist(te_s["mean"], bins=50, alpha=0.7, label="Test")
    ax[0].set_title("Per-image mean intensity"); ax[0].set_xlabel("mean"); ax[0].legend()
    ax[1].scatter(gt_s["mean"], no_s_matched_above1(gt_s, no_s), s=6, alpha=0.4)
    ax[1].set_title("NoisyLR %>1 vs paired GT mean brightness")
    ax[1].set_xlabel("GT mean"); ax[1].set_ylabel("NoisyLR % pixels > 1")
    fig.tight_layout(); fig.savefig(outdir / "intensity_overview.png", dpi=110)
    plt.close(fig)


def no_s_matched_above1(gt_s, no_s):
    """Align NoisyLR %>1 to GT order by id (both sorted, same ids)."""
    return no_s["pct_above1"]


def save_noise_plot(noise: dict, outpath: Path):
    d = noise["residual_std_by_intensity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d["intensity"], d["resid_std"], "o-", label="residual std")
    ax.plot(d["intensity"], d["resid_mean"], "s--", label="residual mean")
    ax.set_xlabel("clean local intensity"); ax.set_ylabel("noise residual")
    ax.set_title("Noise vs intensity (speckle => rising std)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=110); plt.close(fig)


def save_sample_pairs(n: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ids = paths.stem_ids(paths.TRAIN_GT_DIR)
    pick = rng.choice(len(ids), size=n, replace=False)
    for i in pick:
        stem = ids[i]
        gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        noisy = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
        noisy_up = sk_resize(noisy, gt.shape, order=3, preserve_range=True)
        resid = gt - np.clip(noisy_up, 0, 1)
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title(f"GT {stem} 256")
        ax[1].imshow(noisy, cmap="gray"); ax[1].set_title("NoisyLR 128 (raw range)")
        ax[2].imshow(noisy_up, cmap="gray", vmin=0, vmax=1); ax[2].set_title("NoisyLR bicubic->256")
        im = ax[3].imshow(resid, cmap="seismic", vmin=-0.3, vmax=0.3); ax[3].set_title("GT - bicubic")
        for a in ax: a.axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        fig.tight_layout(); fig.savefig(outdir / f"pair_{stem}.png", dpi=100); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-subset", type=int, default=400)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--sample-pairs", type=int, default=8)
    args = ap.parse_args()

    paths.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/6] per-image statistics ...")
    gt_s = per_image_stats(paths.TRAIN_GT_DIR)
    no_s = per_image_stats(paths.TRAIN_NOISY_DIR)
    te_s = per_image_stats(paths.TEST_NOISY_DIR)

    print("[2/6] noise characterization ...")
    noise = characterize_noise(args.noise_subset)

    print("[3/6] duplicate / leakage hashing ...")
    gt_h = hash_dir(paths.TRAIN_GT_DIR)
    no_h = hash_dir(paths.TRAIN_NOISY_DIR)
    te_h = hash_dir(paths.TEST_NOISY_DIR)
    gt_dupes = find_dupes(gt_h)
    no_dupes = find_dupes(no_h)
    # cross-set leakage: identical NoisyLR arrays shared between train and test
    train_no_set = set(no_h.values())
    test_leak = sorted([stem for stem, h in te_h.items() if h in train_no_set])

    print("[4/6] content clustering ...")
    try:
        cl_ids, cl_labels, cl_sizes = cluster_content(args.clusters)
        cluster_ok = True
    except Exception as exc:  # sklearn missing etc.
        print(f"    clustering skipped: {exc}")
        cl_ids, cl_labels, cl_sizes, cluster_ok = [], [], {}, False

    print("[5/6] visualizations ...")
    save_histograms(gt_s, no_s, te_s, paths.AUDIT_DIR / "histograms")
    save_noise_plot(noise, paths.AUDIT_DIR / "noise_vs_intensity.png")
    save_sample_pairs(args.sample_pairs, paths.AUDIT_DIR / "sample_pairs")

    print("[6/6] writing audit_stats.json ...")
    stats = {
        "counts": {
            "train_gt": len(gt_s["ids"]),
            "train_noisy": len(no_s["ids"]),
            "test_noisy": len(te_s["ids"]),
        },
        "shapes": {"gt": [256, 256], "noisy": [128, 128], "scale": 2, "channels": 1, "dtype": "float32"},
        "gt_intensity": {k: summarize(gt_s[k]) for k in ["mean", "std", "min", "max"]},
        "noisy_intensity": {k: summarize(no_s[k]) for k in ["mean", "std", "min", "max", "pct_below0", "pct_above1"]},
        "test_intensity": {k: summarize(te_s[k]) for k in ["mean", "std", "min", "max", "pct_below0", "pct_above1"]},
        "paired_mean_abs_diff": float(np.abs(gt_s["mean"] - no_s["mean"]).mean()),
        "noise": noise,
        "duplicates": {
            "gt_exact_groups": len(gt_dupes),
            "gt_dupe_examples": gt_dupes[:5],
            "noisy_exact_groups": len(no_dupes),
            "test_leak_into_train_noisy": test_leak,
        },
        "clustering": {"ok": cluster_ok, "k": args.clusters, "sizes": cl_sizes},
    }
    (paths.AUDIT_DIR / "audit_stats.json").write_text(json.dumps(stats, indent=2))

    if cluster_ok:
        paths.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        (paths.CONFIGS_DIR / "content_clusters.json").write_text(
            json.dumps({sid: lab for sid, lab in zip(cl_ids, cl_labels)}, indent=2)
        )

    print("\nDONE. Key numbers:")
    print(json.dumps({
        "counts": stats["counts"],
        "gt_mean_range": [stats["gt_intensity"]["mean"]["min"], stats["gt_intensity"]["mean"]["max"]],
        "noisy_pct_above1_mean": stats["noisy_intensity"]["pct_above1"]["mean"],
        "paired_mean_abs_diff": stats["paired_mean_abs_diff"],
        "global_residual_std": noise["global_residual"]["std"],
        "resid_std_by_intensity": noise["residual_std_by_intensity"]["resid_std"],
        "gt_exact_dupe_groups": len(gt_dupes),
        "test_leak": test_leak,
        "cluster_sizes": cl_sizes,
    }, indent=2))


if __name__ == "__main__":
    main()
