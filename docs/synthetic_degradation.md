# Synthetic Degradation Simulator (Phase 9)

Generates synthetic degraded/GT training pairs from the official clean GT images, so we
can (later) test whether extra synthetic data improves the model. Standalone and
reproducible; **off by default** — it never affects the standard competition training
(`configs/nafnet_sr_16gb.yaml` has no `synthetic` block).

Code: `src/augmentation/degradation.py` (core), `src/augmentation/synthetic_dataset.py`
(training wrapper), `scripts/generate_synthetic.py` (offline generator).

---

## What it does — the pipeline

Input: a clean GT `(256, 256)` float in `[0, 1]`. Output: a synthetic NoisyLR
`(128, 128)` float32 that may exceed `[0, 1]` (like the real data). Three mechanisms are
applied in a **configurable order** (default `[downsample, speckle, gaussian]`):

1. **Downsample** — `skimage` resize by `scale` (default 2×), bicubic + anti-aliasing,
   `preserve_range` (no clipping).
2. **Speckle (multiplicative)** — `out = I + I · n`, with `n ~ N(0, (speckle_sigma·sev)²)`.
   Signal-dependent: bright regions get more noise and overshoot above 1 — matching the
   real data's fingerprint.
3. **Additive Gaussian** — `out += N(0, (gauss_sigma·sev)²)`; a signal-independent floor.

`sev` is a per-call severity multiplier (fixed, or sampled from `severity_range` for
mixed-severity augmentation). The GT is copied first and **never modified**; the output is
**not clipped** by default (set `clip_output: true` to force `[0, 1]`).

Why the default order applies noise *after* downsampling: our σ values were **measured at
LR** (post-downsample). Applying noise at LR reproduces those LR statistics. Applying noise
at HR then downsampling averages it and yields a smaller effective LR σ — valid, but a
different regime. Order is fully configurable so both can be explored.

---

## Measured observations vs assumptions

**Measured from our training data** (`docs/dataset_report.md`, `scripts/audit_dataset.py`):
- Downsampling factor is exactly **2×** (256→128).
- Noise is **zero-mean** and its residual std **rises ~linearly with intensity** —
  multiplicative speckle plus an additive floor.
- Fitted defaults: **`speckle_sigma ≈ 0.15`** (residual-std slope; real is ~0.15–0.19, so
  0.15 is a conservative low end) and **`gauss_sigma ≈ 0.013`** (residual std as intensity→0).

**Assumptions / unknowns (NOT claimed to match the hidden KLA pipeline):**
- The **order** the three mechanisms were actually applied in is undisclosed — our default
  is one reasonable choice, not a claim.
- The **downsampling kernel** KLA used is unknown; we use bicubic+anti-aliasing.
- Exact per-image noise parameters and any correlation structure are unknown; we model
  pixel-independent noise.

> The simulator is a **calibrated approximation for data augmentation**, not a
> reconstruction of the official degradation. We only assert its statistics resemble the
> observed training data.

### Calibration smoke (measured vs synthetic, residual std by intensity)

| intensity | 0.05 | 0.25 | 0.45 | 0.65 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|
| **measured (real)** | 0.015 | 0.052 | 0.085 | 0.113 | 0.143 | 0.157 |
| **synthetic (default)** | 0.016 | 0.040 | 0.069 | 0.099 | 0.128 | 0.140 |

Trend and dark-end floor match; the synthetic is intentionally slightly milder at the
bright end (raise `speckle_sigma` toward ~0.16–0.18 to match more tightly).

---

## Configuration parameters (`configs/degradation_default.yaml`)

| Param | Default | Meaning |
|---|---|---|
| `scale` | 2 | downsampling factor |
| `order` | `[downsample, speckle, gaussian]` | sequence of mechanisms |
| `speckle_sigma` | 0.15 | multiplicative-noise std coefficient (·intensity) |
| `gauss_sigma` | 0.013 | additive-noise std |
| `severity` | 1.0 | global σ multiplier |
| `severity_range` | `[0.7, 1.3]` | per-sample severity ~ U(lo,hi); `null` = fixed |
| `downsample_order` | 3 | interpolation (3 = bicubic) |
| `anti_aliasing` | true | anti-alias on downsample |
| `clip_output` | false | keep out-of-`[0,1]` values (like real NoisyLR) |

---

## Tests (`tests/test_degradation.py`)

`output_dimensions` (256→128, 4× →64) · `value_handling` (float32; overshoot >1 when
unclipped; clipped path stays in `[0,1]`) · `deterministic_with_fixed_seed` (same seed ⇒
identical, different seed ⇒ different) · `degradation_order_is_respected` (LR-noise std >
HR-then-downsample std) · `gt_is_not_corrupted` (input array unchanged after repeated
calls) · `config_roundtrip`.

---

## Reproducibility & the future ablation

Every synthetic sample is determined by `(seed, item_index)` via `make_rng`, so a run is
reproducible from its config + seed. **No synthetic results are added to
`experiments/index.csv` until a real ablation is run.**

Planned Phase 9 ablation (Experiment A vs B), to run **after** the first real NAFNet result:

```bash
# A — official pairs only (baseline for the ablation)
python train.py --config configs/nafnet_sr_16gb.yaml

# B — official + on-the-fly synthetic pairs (validation still uses real pairs only)
python train.py --config configs/nafnet_sr_synthetic.yaml
```

Keep synthetic augmentation only if B measurably beats A on IID and/or OOD validation.
