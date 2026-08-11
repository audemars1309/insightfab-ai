# InsightFab AI — AI-Based Restoration of Degraded Semiconductor Inspection Images

**SEMICON India Hackathon 2026 · KLA Problem Statement**
*Restore degraded (noisy, low-resolution) semiconductor inspection images to clean, full-resolution ground truth.*

> Tagline: **Beyond Image Restoration. Towards Inspection Intelligence.**

This repository is the **competition solution** (Path A): dataset audit, training, a
standalone inference script, benchmarking, and reproducible experiment tracking.
An optional inspection-intelligence product layer (InsightFab) is planned on top of the
same restoration engine but is **not** part of the competition inference path.

---

## Task & data contract

The task is **joint 2× super-resolution + denoising** under three degradations
(speckle noise, additive Gaussian noise, downsampling) applied in an undisclosed order.

| | Input (NoisyLR) | Output (restored) |
|---|---|---|
| Format | `.npy`, float32, shape `(128, 128)` | `.npy`, float32, shape `(256, 256)` |
| Range | may extend slightly outside `[0, 1]` (intentional) | clipped to `[0, 1]` |
| Naming | `<id>.npy` | **same** `<id>.npy` |

**Range policy (correctness-critical):** NoisyLR inputs are **never clipped on load**;
restored outputs **are clipped to `[0, 1]` on save**, because KLA scores the saved array
directly against `[0, 1]` GT and never renormalizes it. See `src/utils/npy_io.py`.

Full measured dataset analysis (counts, statistics, noise characterization, duplicate/leakage
checks, splits): [`docs/dataset_report.md`](docs/dataset_report.md).

**Key measured degradation finding:** the noise is zero-bias and its residual std rises
with signal intensity (0.015 → 0.157 dark→bright) — i.e. **multiplicative speckle
(σ ≈ 0.15–0.19·intensity) on an additive Gaussian floor (σ ≈ 0.012–0.015)**. These
measured ranges calibrate the synthetic-degradation simulator.

---

## Environment setup

Requires Python 3.11 and (for training/GPU inference) an NVIDIA GPU.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If your CUDA differs, install torch to match, e.g. for CUDA 11.8:
#   pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

Place the dataset so the layout is:

```
data/train/GT/*.npy        data/train/NoisyLR/*.npy        data/test/NoisyLR/*.npy
```

Or stage them automatically (excludes macOS junk, idempotent):

```bash
python scripts/setup_data.py --src semicon_dataset   # dir with train.zip + Test_NoisyLR.zip
```

---

## Reproduce the pipeline

```bash
# 1. Integrity check (shapes, dtypes, pairing, out-of-range accounting)
python scripts/validate_data.py

# 2. Full dataset audit -> results/dataset_audit/ + configs/content_clusters.json
python scripts/audit_dataset.py

# 3. Reproducible, leakage-free split -> configs/split.json (IID + OOD)
python scripts/build_split.py --seed 42

# 4. Baselines
python scripts/eval_bicubic.py                 # no-learning floor
python scripts/train_baseline_cnn.py --epochs 20   # learned floor

# 4b. Train the competition model (config-driven; 16 GB GPU — see docs/kaggle_colab_setup.md)
python train.py --config configs/nafnet_sr_16gb.yaml
#   resume:  python train.py --config configs/nafnet_sr_16gb.yaml --resume weights/<exp_id>/resume.pt
#   ship:    cp weights/<exp_id>/best.pt weights/final.pt

# 5. Standalone competition inference (evaluator-facing contract)
python inference.py --input_dir <dir_of_noisy_npy> --output_dir <out_dir> \
    --checkpoint weights/final.pt

# 6. End-to-end runtime benchmark (I/O + transfers + model + save)
python benchmark.py --input_dir data/test/NoisyLR --checkpoint weights/final.pt --batch 16
```

`inference.py` requires **no source edits, notebook cells, or hardcoded paths** — only
`--input_dir` and `--output_dir` (plus an optional `--checkpoint`).

---

## Results (measured on the held-out validation splits)

Metrics: PSNR ↑, SSIM ↑, LPIPS ↓. Live numbers are tracked in
[`experiments/index.csv`](experiments/index.csv); each run also writes a full JSON record.

| Method | IID PSNR | IID SSIM | IID LPIPS | OOD PSNR | OOD SSIM | OOD LPIPS |
|---|---|---|---|---|---|---|
| Bicubic ×2 (floor) | 22.86 | 0.537 | 0.452 | 21.47 | 0.390 | 0.561 |
| SmallSRCNN (learned baseline) | 27.69 | 0.753 | 0.296 | 28.68 | 0.677 | 0.427 |
| **Final model** | _TBD_ | | | | | |

The learned baseline improves IID by **+4.8 dB PSNR / +0.22 SSIM / −0.16 LPIPS** over bicubic.
Note the OOD holdout is a *content shift* (a held-out image cluster): its absolute PSNR can
run higher than IID because PSNR depends on image content/brightness, but the true
generalization gap is visible in **SSIM (0.753→0.677) and LPIPS (0.296→0.427)**, which both
degrade OOD. We therefore judge generalization on SSIM/LPIPS, not PSNR alone.

**Runtime (end-to-end, `benchmark.py`, warmup excluded):** on a local **GTX 1650**, batch 16,
fp16 — **46.2 ms/image, 21.7 img/s**; the model is 96.8% of the time, I/O+transfers+save ≈0.6 s
total over 400 images; peak GPU memory **264 MB** (large headroom). KLA benchmarks on an H100,
where throughput will be substantially higher.

---

## Repository layout

```
inference.py            # standalone competition inference (--input_dir/--output_dir)
benchmark.py            # end-to-end runtime benchmark
requirements.txt
configs/                # split.json, content_clusters.json, model configs
src/
  data/                 # paired dataset + split loading
  models/               # restoration architectures
  losses/               # Charbonnier (+ future structural/perceptual)
  evaluation/           # PSNR / SSIM / LPIPS
  inference/            # shared Restorer + checkpoint format
  utils/                # npy I/O (range policy), paths, experiment tracking
scripts/                # validate_data, audit_dataset, build_split, train_*, eval_*
docs/                   # dataset_report.md, external_resources.md
results/dataset_audit/  # measured figures + audit_stats.json
experiments/            # per-run JSON records + index.csv
weights/                # checkpoints (not committed)
```

---

## Reproducibility

Fixed seeds; split saved to `configs/split.json`; every training run writes a timestamped
experiment record (architecture, params, hyperparameters, metrics, runtime, checkpoint path)
under `experiments/`. External resources are disclosed in
[`docs/external_resources.md`](docs/external_resources.md).
