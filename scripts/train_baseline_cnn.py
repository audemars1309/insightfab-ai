"""Train the SmallSRCNN learned baseline and record the experiment.

Establishes the learned floor above bicubic and validates the full pipeline
end-to-end (data -> model -> loss -> train -> eval -> checkpoint). Kept short
and small so it runs on a 4 GB local GPU; the competitive model trains on
cloud GPUs with the same code paths.

Usage (defaults are a quick local run):
    python scripts/train_baseline_cnn.py --epochs 20 --batch 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import PairDataset, load_split  # noqa: E402
from src.evaluation import metrics  # noqa: E402
from src.losses.charbonnier import CharbonnierLoss  # noqa: E402
from src.models.baseline_cnn import build_model, count_params  # noqa: E402
from src.inference.restorer import save_checkpoint  # noqa: E402
from src.utils import experiment, paths  # noqa: E402


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, ids, device, lpips_subset=64):
    model.eval()
    recs = []
    for k, stem in enumerate(ids):
        from src.utils.npy_io import load_npy
        lr = load_npy(paths.TRAIN_NOISY_DIR / f"{stem}.npy")
        gt = load_npy(paths.TRAIN_GT_DIR / f"{stem}.npy")
        lr_t = torch.from_numpy(lr.astype(np.float32))[None, None].to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred = model(lr_t)
        pred = pred[0, 0].float().cpu().numpy()
        use_lpips = k < lpips_subset
        recs.append(metrics.evaluate_pair(pred, gt, use_lpips=use_lpips, device=device))
    # aggregate; lpips only over the subset that computed it
    out = {
        "psnr": float(np.mean([r["psnr"] for r in recs])),
        "ssim": float(np.mean([r["ssim"] for r in recs])),
        "lpips": float(np.mean([r["lpips"] for r in recs if "lpips" in r])),
        "n": len(ids),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-patch", type=int, default=64)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0, help="0=all; small N for overfit sanity")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"device={device}")

    split = load_split()
    train_ids = split["train"] if not args.limit_train else split["train"][: args.limit_train]
    train_ds = PairDataset(train_ids, train=True, lr_patch=args.lr_patch, seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)

    model = build_model("small_srcnn", channels=args.channels, n_blocks=args.blocks).to(device)
    n_params = count_params(model)
    print(f"model params: {n_params:,}")

    criterion = CharbonnierLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    exp_id = experiment.new_exp_id("baseline_cnn")
    ckpt_path = paths.WEIGHTS_DIR / f"{exp_id}.pt"
    paths.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for lr_t, hr_t, _ in train_ld:
            lr_t = lr_t.to(device, non_blocking=True)
            hr_t = hr_t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                pred = model(lr_t)
                loss = criterion(pred, hr_t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()
        avg = running / max(1, len(train_ld))

        # cheap PSNR/SSIM check each epoch on IID val (no LPIPS for speed)
        iid_quick = evaluate(model, split["val_iid"], device, lpips_subset=0)
        print(f"epoch {epoch:02d}/{args.epochs}  loss={avg:.4f}  "
              f"iid_psnr={iid_quick['psnr']:.3f}  iid_ssim={iid_quick['ssim']:.4f}")
        if iid_quick["psnr"] > best_psnr:
            best_psnr = iid_quick["psnr"]
            save_checkpoint(
                ckpt_path, model, "small_srcnn",
                {"channels": args.channels, "n_blocks": args.blocks},
                meta={"exp_id": exp_id, "epoch": epoch, "iid_psnr": best_psnr},
            )

    train_sec = time.perf_counter() - t0

    # final full evaluation with LPIPS on best checkpoint
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    print("final evaluation (best checkpoint) ...")
    val_iid = evaluate(model, split["val_iid"], device, lpips_subset=len(split["val_iid"]))
    val_ood = evaluate(model, split["val_ood"], device, lpips_subset=len(split["val_ood"]))

    record = {
        "exp_id": exp_id,
        "model": "small_srcnn",
        "params": n_params,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr_patch": args.lr_patch,
        "channels": args.channels,
        "blocks": args.blocks,
        "loss": "charbonnier",
        "lr": args.lr,
        "device": device,
        "train_sec": round(train_sec, 1),
        "checkpoint": str(ckpt_path.relative_to(paths.REPO_ROOT)),
        "val_iid": val_iid,
        "val_ood": val_ood,
    }
    experiment.save_record(record)
    print("\n=== RESULT ===")
    print(f"IID  PSNR {val_iid['psnr']:.3f}  SSIM {val_iid['ssim']:.4f}  LPIPS {val_iid['lpips']:.4f}")
    print(f"OOD  PSNR {val_ood['psnr']:.3f}  SSIM {val_ood['ssim']:.4f}  LPIPS {val_ood['lpips']:.4f}")
    print(f"train_sec {train_sec:.1f}  checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
