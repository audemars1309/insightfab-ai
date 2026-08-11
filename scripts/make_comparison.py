"""Phase 13 — visual comparison + failure analysis.

For representative and worst-case validation images, renders a side-by-side panel:

    NoisyLR (input) | Bicubic | SmallCNN | NAFNet-SR | GT | error map (GT - NAFNet)

with per-method PSNR/SSIM/LPIPS in the titles, and writes a metrics summary.
Uses the EXISTING inference restorer + metrics — no competing implementation.

Failure cases are selected objectively as the lowest-NAFNet-PSNR validation images.

Usage (after the real NAFNet checkpoint is available):
    python scripts/make_comparison.py \
        --nafnet weights/final.pt --smallcnn weights/baseline_cnn_20260811_173144.pt \
        --n-good 4 --n-fail 4 --out results/comparisons
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
from src.evaluation import metrics  # noqa: E402
from src.inference.restorer import Restorer, load_restorer  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.npy_io import GT_SHAPE, load_npy  # noqa: E402


def bicubic_up(noisy):
    return sk_resize(noisy, GT_SHAPE, order=3, preserve_range=True).astype(np.float32)


def restore_one(restorer, lr):
    return restorer.restore_batch(lr[None].astype(np.float32))[0]


def method_metrics(pred, gt, device):
    return metrics.evaluate_pair(pred, gt, use_lpips=True, device=device)


def render_panel(stem, lr, gt, preds: dict, out_path, device):
    disp_lr = np.clip(bicubic_up(lr), 0, 1)
    err = gt - np.clip(preds["NAFNet-SR"], 0, 1)
    cols = [("NoisyLR (input)", disp_lr, None),
            ("Bicubic", np.clip(preds["Bicubic"], 0, 1), preds["_m"]["Bicubic"]),
            ("SmallCNN", np.clip(preds["SmallCNN"], 0, 1), preds["_m"]["SmallCNN"]),
            ("NAFNet-SR", np.clip(preds["NAFNet-SR"], 0, 1), preds["_m"]["NAFNet-SR"]),
            ("GT", gt, None)]
    fig, ax = plt.subplots(1, 6, figsize=(22, 4))
    for i, (name, img, m) in enumerate(cols):
        ax[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        title = name if m is None else f"{name}\nPSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f}\nLPIPS {m['lpips']:.3f}"
        ax[i].set_title(title, fontsize=9)
        ax[i].axis("off")
    im = ax[5].imshow(err, cmap="seismic", vmin=-0.3, vmax=0.3)
    ax[5].set_title("Error (GT - NAFNet)", fontsize=9); ax[5].axis("off")
    fig.colorbar(im, ax=ax[5], fraction=0.046)
    fig.suptitle(f"id {stem}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build_for(stem, restorers, device):
    lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
    gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
    preds = {
        "Bicubic": bicubic_up(lr),
        "SmallCNN": restore_one(restorers["SmallCNN"], lr),
        "NAFNet-SR": restore_one(restorers["NAFNet-SR"], lr),
    }
    preds["_m"] = {k: method_metrics(preds[k], gt, device) for k in ["Bicubic", "SmallCNN", "NAFNet-SR"]}
    return lr, gt, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nafnet", required=True, type=Path)
    ap.add_argument("--smallcnn", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "comparisons")
    ap.add_argument("--n-good", type=int, default=4)
    ap.add_argument("--n-fail", type=int, default=4)
    ap.add_argument("--ids", nargs="*", default=None, help="explicit ids (skips selection)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    naf_model, _ = load_restorer(args.nafnet, device=device)
    scn_model, _ = load_restorer(args.smallcnn, device=device)
    restorers = {"NAFNet-SR": Restorer(naf_model, device), "SmallCNN": Restorer(scn_model, device)}

    args.out.mkdir(parents=True, exist_ok=True)
    split = load_split()

    if args.ids:
        good_ids, fail_ids = args.ids, []
    else:
        # Rank all validation images by NAFNet PSNR to pick representative + failures.
        val_ids = split["val_iid"] + split["val_ood"]
        scored = []
        for stem in val_ids:
            lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
            gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
            scored.append((metrics.psnr(restore_one(restorers["NAFNet-SR"], lr), gt), stem))
        scored.sort()
        fail_ids = [s for _, s in scored[: args.n_fail]]
        # "good/representative" = around the median
        mid = len(scored) // 2
        good_ids = [s for _, s in scored[mid: mid + args.n_good]]

    summary = {"nafnet": str(args.nafnet), "smallcnn": str(args.smallcnn),
               "representative": [], "failures": []}
    for group, ids in [("representative", good_ids), ("failures", fail_ids)]:
        gdir = args.out / group
        gdir.mkdir(parents=True, exist_ok=True)
        for stem in ids:
            lr, gt, preds = build_for(stem, restorers, device)
            render_panel(stem, lr, gt, preds, gdir / f"{stem}.png", device)
            summary[group].append({"id": stem, "metrics": {k: preds["_m"][k] for k in preds["_m"]}})

    (args.out / "comparison_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote panels to {args.out} ({len(good_ids)} representative, {len(fail_ids)} failures)")
    print(f"metrics -> {args.out / 'comparison_metrics.json'}")


if __name__ == "__main__":
    main()
