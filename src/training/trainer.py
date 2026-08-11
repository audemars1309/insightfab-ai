"""Config-driven training engine for the KLA restoration models.

Features (all standard, no over-engineering):
* mixed precision (AMP) with GradScaler
* EMA of weights (a reliable quality boost for restoration); EMA weights are
  what we validate and ship
* best / last checkpointing + full resume (model, EMA, optimizer, scheduler,
  scaler, epoch, best score)
* periodic IID + OOD validation (PSNR/SSIM always, LPIPS on a subset)
* structured experiment record written via src.utils.experiment
* fixed seeds and optional cudnn determinism

Checkpoints use the shared self-describing format (src.inference.restorer) so
inference.py / benchmark.py load them without knowing the architecture.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PairDataset, load_split
from src.evaluation.validate import evaluate_model
from src.inference.restorer import save_checkpoint
from src.losses.combined import build_loss
from src.models.registry import build_model, count_params
from src.utils import experiment, paths


def set_seed(seed: int, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def _composite(m: dict) -> float:
    """Model-selection score. KLA's exact weighting is undisclosed, so we use a
    transparent proxy that rewards PSNR & SSIM and penalizes LPIPS. Reported
    metrics remain the source of truth; this only picks 'best' for checkpointing.
    """
    lp = m.get("lpips")
    return m["psnr"] + 20.0 * m["ssim"] - (10.0 * lp if lp is not None else 0.0)


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg["seed"], cfg.get("deterministic", False))

        # data
        split = load_split(cfg.get("split_path"))
        tr = cfg["train"]
        train_ids = split["train"]
        if tr.get("limit_train"):
            train_ids = train_ids[: tr["limit_train"]]
        self.val_iid_ids = split["val_iid"]
        self.val_ood_ids = split["val_ood"]
        # Default path: official real pairs. The synthetic block is opt-in (Phase 9
        # ablation); absent it, behavior is byte-identical to before. Validation
        # always uses real pairs (val_iid/val_ood), never synthetic.
        syn = cfg.get("synthetic") or {}
        if syn.get("enabled"):
            from src.augmentation.synthetic_dataset import SyntheticMixDataset
            train_ds = SyntheticMixDataset(
                train_ids, lr_patch=tr["lr_patch"], seed=cfg["seed"],
                degrade_cfg=syn.get("degradation"), synth_prob=syn.get("synth_prob", 0.5),
            )
            print(f"[trainer] synthetic augmentation ON (synth_prob={syn.get('synth_prob', 0.5)})")
        else:
            train_ds = PairDataset(train_ids, train=True, lr_patch=tr["lr_patch"], seed=cfg["seed"])
        self.train_ld = DataLoader(
            train_ds, batch_size=tr["batch"], shuffle=True,
            num_workers=tr.get("workers", 4), pin_memory=(self.device == "cuda"),
            drop_last=True, persistent_workers=tr.get("workers", 4) > 0,
        )

        # model / loss / optim
        self.model = build_model(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(self.device)
        self.n_params = count_params(self.model)
        self.criterion = build_loss(cfg.get("loss", {})).to(self.device)
        opt_cfg = cfg["optim"]
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.9))),
        )
        self.epochs = tr["epochs"]
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.epochs, eta_min=opt_cfg.get("eta_min", 1e-6))
        self.use_amp = cfg.get("amp", True) and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = tr.get("grad_clip", 0.0)

        # EMA
        self.ema = EMA(self.model, cfg.get("ema_decay", 0.999)) if cfg.get("use_ema", True) else None

        # bookkeeping
        self.exp_id = cfg.get("exp_id") or experiment.new_exp_id(cfg["model"]["name"])
        self.out_dir = paths.WEIGHTS_DIR / self.exp_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_every = tr.get("val_every", 1)
        self.lpips_n = cfg.get("val", {}).get("lpips_n", 64)
        self.start_epoch = 1
        self.best_score = -1e9

        print(f"[trainer] exp={self.exp_id} model={cfg['model']['name']} "
              f"params={self.n_params:,} device={self.device} amp={self.use_amp} "
              f"ema={self.ema is not None} batch={tr['batch']} patch={tr['lr_patch']}")

    # --- checkpointing -----------------------------------------------------
    def _eval_model(self):
        m = self.ema.shadow if self.ema else self.model
        iid = evaluate_model(m, self.val_iid_ids, self.device, self.lpips_n, self.use_amp)
        ood = evaluate_model(m, self.val_ood_ids, self.device, self.lpips_n, self.use_amp)
        return iid, ood

    def _save_ship(self, name: str, meta: dict):
        """Save an inference-ready checkpoint (EMA weights if enabled)."""
        m = self.ema.shadow if self.ema else self.model
        save_checkpoint(self.out_dir / name, m, self.cfg["model"]["name"],
                        self.cfg["model"].get("kwargs", {}), meta=meta)

    def _save_resume(self, epoch: int):
        ck = {
            "epoch": epoch, "best_score": self.best_score,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
            "cfg": self.cfg,
            # RNG state so a resumed session continues the same stochastic stream.
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
        }
        torch.save(ck, self.out_dir / "resume.pt")

    def maybe_resume(self, path):
        if not path or not Path(path).exists():
            return
        # weights_only=False: resume.pt is our own trusted checkpoint and holds
        # optimizer/scheduler/RNG state (not just tensors). Shippable checkpoints
        # (best/last/final) contain no such objects and load with the safe default.
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.scaler.load_state_dict(ck["scaler"])
        if self.ema and ck.get("ema"):
            self.ema.load_state_dict(ck["ema"])
        rng = ck.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception:
                    pass  # GPU-count mismatch across sessions; safe to skip
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
        self.start_epoch = ck["epoch"] + 1
        self.best_score = ck["best_score"]
        print(f"[trainer] resumed from {path} at epoch {self.start_epoch}")

    # --- training ----------------------------------------------------------
    def fit(self):
        t0 = time.perf_counter()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            running = 0.0
            for lr_t, hr_t, _ in self.train_ld:
                lr_t = lr_t.to(self.device, non_blocking=True)
                hr_t = hr_t.to(self.device, non_blocking=True)
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    pred = self.model(lr_t)
                    loss, _ = self.criterion(pred, hr_t)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                if self.ema:
                    self.ema.update(self.model)
                running += loss.item()
            self.sched.step()
            avg = running / max(1, len(self.train_ld))

            msg = f"epoch {epoch:03d}/{self.epochs}  loss={avg:.4f}  lr={self.sched.get_last_lr()[0]:.2e}"
            if epoch % self.val_every == 0 or epoch == self.epochs:
                iid, ood = self._eval_model()
                score = _composite(iid)
                msg += (f"  IID[psnr={iid['psnr']:.3f} ssim={iid['ssim']:.4f} "
                        f"lpips={iid['lpips']:.4f}]  OOD[psnr={ood['psnr']:.3f} ssim={ood['ssim']:.4f}]")
                if score > self.best_score:
                    self.best_score = score
                    self._save_ship("best.pt", {"exp_id": self.exp_id, "epoch": epoch,
                                                "val_iid": iid, "val_ood": ood})
                    msg += "  *best"
            print(msg, flush=True)
            self._save_ship("last.pt", {"exp_id": self.exp_id, "epoch": epoch})
            self._save_resume(epoch)

        train_sec = time.perf_counter() - t0

        # Final evaluation of the best (shipped) checkpoint with a larger LPIPS set.
        from src.inference.restorer import load_restorer
        best_model, _ = load_restorer(self.out_dir / "best.pt", device=self.device)
        iid = evaluate_model(best_model, self.val_iid_ids, self.device,
                             lpips_n=len(self.val_iid_ids), fp16=self.use_amp)
        ood = evaluate_model(best_model, self.val_ood_ids, self.device,
                             lpips_n=len(self.val_ood_ids), fp16=self.use_amp)

        record = {
            "exp_id": self.exp_id,
            "model": self.cfg["model"]["name"],
            "model_kwargs": self.cfg["model"].get("kwargs", {}),
            "params": self.n_params,
            "seed": self.cfg["seed"],
            "epochs": self.epochs,
            "batch": self.cfg["train"]["batch"],
            "lr_patch": self.cfg["train"]["lr_patch"],
            "loss": self.cfg.get("loss", {}),
            "lr": self.cfg["optim"]["lr"],
            "ema": self.ema is not None,
            "device": self.device,
            "amp": self.use_amp,
            "train_sec": round(train_sec, 1),
            "checkpoint": str((self.out_dir / "best.pt").relative_to(paths.REPO_ROOT)),
            "val_iid": iid,
            "val_ood": ood,
        }
        experiment.save_record(record)
        print("\n=== FINAL (best checkpoint) ===")
        print(f"IID  PSNR {iid['psnr']:.3f}  SSIM {iid['ssim']:.4f}  LPIPS {iid['lpips']:.4f}")
        print(f"OOD  PSNR {ood['psnr']:.3f}  SSIM {ood['ssim']:.4f}  LPIPS {ood['lpips']:.4f}")
        print(f"train_sec {train_sec:.1f}  best -> {self.out_dir / 'best.pt'}")
        return record
