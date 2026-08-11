# Dataset Report — KLA Semiconductor Image Restoration

**Project:** InsightFab AI · **Task:** paired restoration of degraded (NoisyLR → GT) semiconductor inspection images
**Source:** local `semicon_dataset/{train.zip, Test_NoisyLR.zip}` (official KLA dataset)
**Status:** all numbers below are **measured** by `scripts/validate_data.py` and `scripts/audit_dataset.py`. Nothing here is assumed.
**Reproduce:** `python scripts/validate_data.py` then `python scripts/audit_dataset.py`
Raw outputs: [`results/dataset_audit/audit_stats.json`](../results/dataset_audit/audit_stats.json), figures under `results/dataset_audit/`.

---

## 1–4. Counts

| Set | Files | Role |
|---|---|---|
| `train/GT` | **3200** | clean targets, 256×256 |
| `train/NoisyLR` | **3200** | degraded inputs, 128×128 |
| `test/NoisyLR` | **400** | degraded-only inputs, 128×128 (GT withheld by KLA) |

The archives also contained macOS `__MACOSX/._*` resource-fork files and `.DS_Store`; these are **not** data and are excluded on ingest.

## 5–7. Dimensions, channels, dtype

| | GT | NoisyLR | Test NoisyLR |
|---|---|---|---|
| Shape | `(256, 256)` | `(128, 128)` | `(128, 128)` |
| Channels | 1 (grayscale, 2-D array) | 1 | 1 |
| dtype | `float32` | `float32` | `float32` |

**Downsampling factor = exactly 2×** (128→256). The task is therefore **joint 2× super-resolution + denoising**, not pure denoising. Every file is uniform in size — there is no size distribution to model (item 12 is trivially a single point).

## 8–9. Value ranges

- **GT is strictly in [0, 1]** for every one of the 3200 files (measured per-image min = 0.0, max = 1.0 across the whole set). This matches the official normalization rule.
- **NoisyLR extends outside [0, 1] intentionally.** Whole-dataset pixel fractions:
  - train: **0.28 % below 0**, **3.11 % above 1**
  - test: **0.66 % below 0**, **3.08 % above 1**
- Per-image extremes go as far as NoisyLR **min ≈ −0.28** and **max ≈ 2.16**. The above-1 fraction is highly image-dependent (per-image up to **44 %** for very bright images).

**Engineering rule enforced in code:** do **not** clip NoisyLR on load (`src/utils/npy_io.load_npy`); clip the restored output to [0, 1] on save (`save_restored`), because the scorer compares directly to [0, 1] GT and never renormalizes our file.

## 10–11. Intensity distributions & statistics

| Statistic (per-image, averaged over set) | GT | NoisyLR | Test |
|---|---|---|---|
| mean intensity | 0.434 | 0.434 | 0.443 |
| mean of per-image std (contrast) | 0.188 | 0.206 | 0.220 |
| per-image mean spans | 0.016 – 0.959 | 0.016 – 0.959 | 0.033 – 0.905 |

Brightness varies widely across the set (dark to bright scenes), so the model must be brightness-robust. NoisyLR per-image std is systematically **higher** than GT (noise inflates variance) while the **mean is preserved**. Test-set statistics sit in the same regime as train — consistent with the official note that test noise levels vary only *within a similar range*. Figure: [`histograms/intensity_overview.png`](../results/dataset_audit/histograms/intensity_overview.png).

## 12. Image-size distribution
Single-valued: all GT 256×256, all NoisyLR 128×128. No variation.

## 13–14. Representative visualizations & difference maps
8 side-by-side panels (GT · raw NoisyLR · bicubic→256 · residual) under [`sample_pairs/`](../results/dataset_audit/sample_pairs). The residual panels show spatially-white, high-frequency noise with no structured pattern, i.e. the degradation is dominated by pixel-wise noise + resolution loss rather than blur/warp artifacts.

## 15. Noise observations (degradation characterization) — key finding

We isolated the *noise* from the *resolution gap* by bicubic-downsampling each GT to 128×128 and taking the residual `NoisyLR − downsample(GT)` over 400 pairs. Figure: [`noise_vs_intensity.png`](../results/dataset_audit/noise_vs_intensity.png).

- **Residual mean ≈ 0** at every intensity (−0.0015 … +0.0017) and globally (−4.4e-05). The noise is **zero-bias / mean-preserving** — confirmed independently by the paired per-image mean difference of only **0.00056**.
- **Residual std rises monotonically with clean intensity**, near-linearly:

  | clean intensity | 0.05 | 0.25 | 0.45 | 0.65 | 0.85 | 0.95 |
  |---|---|---|---|---|---|---|
  | residual std | 0.015 | 0.052 | 0.085 | 0.113 | 0.143 | 0.157 |

**Interpretation (evidence-based):** a flat residual std would indicate pure additive Gaussian noise; a std that grows with signal is the fingerprint of **multiplicative speckle**. The data shows both:
- an **additive Gaussian floor**, σ_add ≈ **0.012–0.015** (the nonzero std as intensity → 0), plus
- a **signal-proportional speckle** term, σ_speckle ≈ **0.15–0.19 × intensity**.

This is a direct, quantified match to the two official noise mechanisms (speckle + additive Gaussian). These measured ranges **calibrate the synthetic-degradation simulator** (Phase 9) so generated pairs match the real noise regime rather than guessed parameters. Global residual std ≈ **0.088**.

## 16. Resolution degradation observations
Fixed 2× downsampling. The bicubic-up panels recover coarse structure but leave the fine, high-frequency detail that the learned model must reconstruct — this is where SR capacity (not just denoising) earns its metrics.

## 17. Pairing validation
GT and NoisyLR filename stems match **exactly**: 3200 matched, **0 GT-only, 0 NoisyLR-only**. Paired mean-intensity difference of 0.00056 confirms each pair is the same underlying image.

## 18. Duplicate / leakage analysis
- **Exact-duplicate GT groups: 0** (MD5 over raw float32 bytes). No within-train duplication.
- **Exact-duplicate NoisyLR groups: 0.**
- **Test → train leakage: none.** No test NoisyLR array is byte-identical to any train NoisyLR array.
- **Filename caution:** test IDs are `000000–000399` and *reuse* train's numbering, but the **content differs** (e.g. test `000267` per-image mean 0.52 vs train `000267` mean 0.245 — a gap noise cannot produce). Filenames are therefore **not** cross-split content identifiers; treat the test set as independent content.

## 19. Image-family / category analysis
No category metadata is provided (flat folders). Unsupervised K-means (k=8) on 16×16 GT thumbnails yields reasonably balanced content clusters (sizes 249–554), i.e. **natural image families exist**. Cluster assignment saved to [`configs/content_clusters.json`](../configs/content_clusters.json) and used to build the OOD split.

| cluster | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| size | 249 | 515 | 343 | 554 | 554 | 264 | 256 | 465 |

## 20. Proposed validation split
Image-level split (a NoisyLR/GT pair is the atom; **no patch is shared** between train and val), fixed seed, saved to disk for exact reproduction:
- **IID validation:** ~10 % (≈320 pairs) sampled *stratified by content cluster* so it mirrors the training distribution.
- Never select the final model on the training set; all model selection uses these held-out sets.

## 21. OOD validation strategy
Hold out **entire content clusters** the model never trains on, approximating the "unfamiliar image content" in the hidden test set. Report PSNR/SSIM/LPIPS **separately for IID and OOD**; the final model is chosen on a quality **and** OOD-generalization **and** throughput trade-off, not IID PSNR alone.

---

### Consequences carried into modeling
1. **Architecture must super-resolve 2×**, not just denoise → needs an upsampling stage (e.g. pixel-shuffle tail).
2. **Handle out-of-[0,1] inputs**: keep raw on input, clip only on output save.
3. **Noise is signal-dependent (speckle) + additive** → the simulator and any noise-aware design target this measured regime.
4. **OOD is a first-class objective** → cluster-holdout validation, not random splits.
5. **Throughput counts end-to-end on H100** → favor efficient architectures and an optimized I/O pipeline.
