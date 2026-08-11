"""Phase 9 — synthetic degradation simulator.

Turns a clean GT image into a synthetic NoisyLR observation that matches the
*measured* degradation regime of the official KLA training data, so we can
create extra training pairs and later ablate their value.

Calibration (see docs/synthetic_degradation.md and docs/dataset_report.md)
-------------------------------------------------------------------------
These numbers are **empirical observations from our training data**, NOT the
disclosed KLA pipeline (order and exact parameters are hidden):

  * multiplicative speckle:   residual std ≈ 0.15 · local_intensity  -> speckle_sigma ≈ 0.15
  * additive Gaussian floor:  residual std ≈ 0.013 at intensity → 0    -> gauss_sigma  ≈ 0.013
  * downsampling factor:      exactly 2×  (256 → 128)

They are the *defaults* and are fully configurable.

Design rules enforced here
--------------------------
* The input GT is never mutated (we operate on a copy).
* NoisyLR output is NOT clipped by default — real NoisyLR legitimately exceeds
  [0, 1] (bright speckle overshoot), and clipping would destroy that.
* Degradation order is explicit and configurable. Because our σ values were
  measured *at LR* (after downsampling), the default order applies noise after
  downsampling so synthetic LR statistics match the measurement. Applying noise
  before downsampling is allowed but yields smaller effective LR noise (averaging).
* Everything is seed-driven and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from skimage.transform import resize as sk_resize

STEPS = ("downsample", "speckle", "gaussian")


@dataclass
class DegradationConfig:
    scale: int = 2
    # Order the three mechanisms are applied in. Default = noise at LR (matches
    # our LR-measured σ). Any permutation/subset of STEPS is accepted.
    order: tuple = ("downsample", "speckle", "gaussian")
    speckle_sigma: float = 0.15          # measured slope of residual-std vs intensity
    gauss_sigma: float = 0.013           # measured additive floor
    severity: float = 1.0                # global multiplier on both σ
    # Per-sample severity multiplier ~ U(lo, hi) for "mixed degradation severity".
    severity_range: tuple | None = None  # e.g. (0.7, 1.3); None = fixed severity
    downsample_order: int = 3            # skimage interpolation (3 = bicubic)
    anti_aliasing: bool = True
    clip_output: bool = False            # keep out-of-[0,1] values like real NoisyLR

    def validate(self):
        for s in self.order:
            if s not in STEPS:
                raise ValueError(f"unknown degradation step '{s}'; allowed: {STEPS}")
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        return self

    @classmethod
    def from_dict(cls, d: dict | None) -> "DegradationConfig":
        d = dict(d or {})
        if "order" in d and d["order"] is not None:
            d["order"] = tuple(d["order"])
        if d.get("severity_range") is not None:
            d["severity_range"] = tuple(d["severity_range"])
        return cls(**d).validate()

    def to_dict(self) -> dict:
        return asdict(self)


class DegradationSimulator:
    """Applies a configured degradation to GT images. Stateless apart from cfg;
    randomness is supplied per call via a NumPy Generator for reproducibility."""

    def __init__(self, cfg: DegradationConfig | dict | None = None):
        if not isinstance(cfg, DegradationConfig):
            cfg = DegradationConfig.from_dict(cfg)
        self.cfg = cfg.validate()

    # --- individual mechanisms --------------------------------------------
    def _downsample(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        out = sk_resize(
            img, (h // self.cfg.scale, w // self.cfg.scale),
            order=self.cfg.downsample_order,
            anti_aliasing=self.cfg.anti_aliasing,
            preserve_range=True,
        )
        return out.astype(np.float32)

    def _speckle(self, img: np.ndarray, rng: np.random.Generator, sev: float) -> np.ndarray:
        # Multiplicative: out = I + I * n, n ~ N(0, (speckle_sigma*sev)^2).
        n = rng.normal(0.0, self.cfg.speckle_sigma * sev, size=img.shape).astype(np.float32)
        return img + img * n

    def _gaussian(self, img: np.ndarray, rng: np.random.Generator, sev: float) -> np.ndarray:
        n = rng.normal(0.0, self.cfg.gauss_sigma * sev, size=img.shape).astype(np.float32)
        return img + n

    # --- full pipeline -----------------------------------------------------
    def degrade(self, gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """GT (H, W) float in [0,1] -> synthetic NoisyLR (H/scale, W/scale) float32.

        The input array is never modified.
        """
        img = np.array(gt, dtype=np.float32, copy=True)  # defensive copy; GT untouched
        sev = self.cfg.severity
        if self.cfg.severity_range is not None:
            sev *= float(rng.uniform(*self.cfg.severity_range))

        for step in self.cfg.order:
            if step == "downsample":
                img = self._downsample(img)
            elif step == "speckle":
                img = self._speckle(img, rng, sev)
            elif step == "gaussian":
                img = self._gaussian(img, rng, sev)

        if self.cfg.clip_output:
            img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float32)


def make_rng(seed: int, index: int = 0) -> np.random.Generator:
    """Deterministic per-item generator: same (seed, index) -> same stream."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))
