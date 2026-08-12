"""A/B comparison + hallucination analysis for the controlled experiment.

Consumes two experiment output dirs (each with history.json and best_*.pt from
the enhanced Trainer) and produces:
  * per-metric peak table (value + epoch) for A and B
  * late-epoch values and post-peak degradation
  * validation curves (PNG) for all 6 metrics, A vs B
  * a metric-by-metric Pareto delta (no invented composite weighting)
  * frequency / hallucination analysis: compares each model's output
    high-frequency energy against GT (added HF => possible hallucination),
    plus residual std and an edge-overshoot (ringing) proxy.

Usage:
    python scripts/compare_ab.py --a weights/nafnet_sr_A_e80 --b weights/nafnet_sr_degaug_B_e80 \
        --a-name baseline --b-name candidate --out results/ab [--no-halluc]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage.transform import resize as sk_resize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_split  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.npy_io import load_npy  # noqa: E402

METRICS = ["iid_psnr", "iid_ssim", "iid_lpips", "ood_psnr", "ood_ssim", "ood_lpips"]
HIGHER_BETTER = {"iid_psnr", "iid_ssim", "ood_psnr", "ood_ssim"}


def load_history(d: Path):
    return json.loads((d / "history.json").read_text())


def peak_of(history, metric):
    vals = [(h[metric], h["epoch"]) for h in history if h.get(metric) is not None]
    if not vals:
        return None, None
    if metric in HIGHER_BETTER:
        return max(vals, key=lambda x: x[0])
    return min(vals, key=lambda x: x[0])


def radial_hf_fraction(img, cutoff=0.5):
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    p = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    prof = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    prof = prof[: min(cy, cx)]
    k = int(len(prof) * cutoff)
    return float(prof[k:].sum() / (prof.sum() + 1e-12))


def peak_table(hist_a, hist_b, name_a, name_b):
    rows = []
    for m in METRICS:
        (va, ea), (vb, eb) = peak_of(hist_a, m), peak_of(hist_b, m)
        better = ("=" if va is None or vb is None else
                  (name_b if ((vb > va) == (m in HIGHER_BETTER) and vb != va) else name_a))
        # post-peak degradation (final minus peak, in the "worse" direction)
        fa, fb = hist_a[-1].get(m), hist_b[-1].get(m)
        da = None if fa is None else (va - fa if m in HIGHER_BETTER else fa - va)
        db = None if fb is None else (vb - fb if m in HIGHER_BETTER else fb - vb)
        rows.append(dict(metric=m, a_peak=va, a_epoch=ea, b_peak=vb, b_epoch=eb,
                         better=better, a_postpeak_drop=da, b_postpeak_drop=db))
    return rows


def plot_curves(hist_a, hist_b, name_a, name_b, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, m in zip(axes.ravel(), METRICS):
        ea = [h["epoch"] for h in hist_a if h.get(m) is not None]
        va = [h[m] for h in hist_a if h.get(m) is not None]
        eb = [h["epoch"] for h in hist_b if h.get(m) is not None]
        vb = [h[m] for h in hist_b if h.get(m) is not None]
        ax.plot(ea, va, "o-", label=name_a)
        ax.plot(eb, vb, "s-", label=name_b)
        ax.set_title(m); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=110)
    plt.close(fig)


def halluc_analysis(ckpt_a, ckpt_b, name_a, name_b, device, n=40):
    """Compare output high-frequency energy vs GT for both models. Added HF
    relative to GT (ratio > 1) indicates possible hallucinated structure."""
    import torch
    from src.inference.restorer import Restorer, load_restorer

    split = load_split()
    rng = np.random.default_rng(0)
    ids = ([split["val_iid"][i] for i in rng.choice(len(split["val_iid"]), min(n, len(split["val_iid"])), replace=False)] +
           [split["val_ood"][i] for i in rng.choice(len(split["val_ood"]), min(n, len(split["val_ood"])), replace=False)])

    def run(ckpt):
        model, _ = load_restorer(ckpt, device=device)
        r = Restorer(model, device=device, fp16=(device == "cuda"))
        gt_hf, out_hf, resid, overshoot = [], [], [], []
        for stem in ids:
            lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
            gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
            pred = np.clip(r.restore_batch(lr[None].astype(np.float32))[0], 0, 1)
            gt_hf.append(radial_hf_fraction(gt))
            out_hf.append(radial_hf_fraction(pred))
            resid.append(float(np.std(pred - gt)))
            # ringing/overshoot proxy: fraction of pixels exceeding local GT range near edges
            overshoot.append(float(np.mean(np.abs(pred - gt) > 0.15)))
        return dict(gt_hf=float(np.mean(gt_hf)), out_hf=float(np.mean(out_hf)),
                    hf_ratio=float(np.mean(out_hf) / (np.mean(gt_hf) + 1e-9)),
                    resid_std=float(np.mean(resid)), overshoot_frac=float(np.mean(overshoot)))

    return {name_a: run(ckpt_a), name_b: run(ckpt_b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-name", default="A_baseline")
    ap.add_argument("--b-name", default="B_candidate")
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "ab")
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint name for hallucination analysis")
    ap.add_argument("--no-halluc", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    hist_a, hist_b = load_history(args.a), load_history(args.b)

    rows = peak_table(hist_a, hist_b, args.a_name, args.b_name)
    plot_curves(hist_a, hist_b, args.a_name, args.b_name, args.out)

    summary = {"a_name": args.a_name, "b_name": args.b_name, "peaks": rows}

    if not args.no_halluc:
        import torch
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ca, cb = args.a / args.ckpt, args.b / args.ckpt
        if ca.exists() and cb.exists():
            summary["hallucination"] = halluc_analysis(ca, cb, args.a_name, args.b_name, dev)
        else:
            summary["hallucination"] = f"checkpoints not found ({ca}, {cb}); skipped"

    (args.out / "ab_summary.json").write_text(json.dumps(summary, indent=2))

    # console report
    print("=" * 78)
    print(f"A/B PEAK COMPARISON   A={args.a_name}   B={args.b_name}")
    print("=" * 78)
    print(f"{'metric':10s} {'A peak':>10s} {'@ep':>4s} {'B peak':>10s} {'@ep':>4s} {'better':>12s}")
    for r in rows:
        print(f"{r['metric']:10s} {r['a_peak']:>10.4f} {str(r['a_epoch']):>4s} "
              f"{r['b_peak']:>10.4f} {str(r['b_epoch']):>4s} {r['better']:>12s}")
    print("\npost-peak drop (peak - final; lower=more stable):")
    for r in rows:
        print(f"  {r['metric']:10s} A={r['a_postpeak_drop']}  B={r['b_postpeak_drop']}")
    if isinstance(summary.get("hallucination"), dict):
        print("\nfrequency / hallucination (output HF vs GT HF; ratio~1 faithful, >>1 = added HF):")
        for name, s in summary["hallucination"].items():
            print(f"  {name:14s} gt_hf={s['gt_hf']:.4f} out_hf={s['out_hf']:.4f} "
                  f"hf_ratio={s['hf_ratio']:.2f} resid_std={s['resid_std']:.4f} "
                  f"overshoot={s['overshoot_frac']:.4f}")
    print(f"\nwrote {args.out/'ab_summary.json'} and {args.out/'validation_curves.png'}")


if __name__ == "__main__":
    main()
