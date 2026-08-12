# InsightFab AI — Restoration of Degraded Semiconductor Inspection Images

Solution for the SEMICON India Hackathon 2026, KLA problem statement
*"AI-Based Restoration of Degraded Images for Semiconductor Inspection."*

The task: take a degraded (noisy, low-resolution) semiconductor inspection image and
reconstruct a clean, full-resolution image as close as possible to the ground truth.

---

## The problem

Each input is a noisy, downsampled image. Three degradations are applied in an
undisclosed order: **speckle noise, additive Gaussian noise, and 2× downsampling**.
The model has to remove the noise and upsample back to the ground-truth resolution.
The hidden test set only provides the degraded inputs; the clean targets are held by KLA
for scoring. So the whole project is really one thing: a good restoration model.

**Input / output contract**

| | Input (NoisyLR) | Output (restored) |
|---|---|---|
| Format | `.npy`, float32, `(128, 128)` | `.npy`, float32, `(256, 256)` |
| Range | may go slightly outside `[0, 1]` (intentional) | clipped to `[0, 1]` |
| Filename | `<id>.npy` | **same** `<id>.npy` |

**Range handling (important):** the noisy inputs are **not** clipped when loaded — some
pixels legitimately sit above 1.0 (bright speckle) or just below 0.0. The restored output
**is** clipped to `[0, 1]` on save, because the scorer compares it directly to `[0, 1]`
ground truth. See `src/utils/npy_io.py`.

---

## Dataset structure

Everything is stored as NumPy `.npy` arrays (single-channel, float32).

```
data/
  train/
    GT/<id>.npy        # 3200 clean images, (256, 256), values in [0, 1]
    NoisyLR/<id>.npy   # 3200 degraded inputs, (128, 128)
  test/
    NoisyLR/<id>.npy   # 400 degraded test inputs, (128, 128)
```

A reproducible, leakage-free split of the 3200 training pairs is saved in
`configs/split.json` (train 2651 / IID-val 293 / OOD-val 256). The OOD-val set is a
held-out content cluster used to check generalization to unfamiliar image content.

The full measured dataset analysis (counts, statistics, noise model, duplicate/leakage
checks) is in [`docs/dataset_report.md`](docs/dataset_report.md). One useful finding: the
noise is zero-mean and its standard deviation grows with pixel intensity
(≈ 0.009 + 0.156 × intensity), i.e. multiplicative speckle on top of an additive Gaussian
floor.

---

## Model

**NAFNet-SR** — a NAFNet restoration backbone (Chen et al., ECCV 2022) with a PixelShuffle
×2 super-resolution tail and a bicubic global skip connection. Single-channel in/out,
2× upscaling. Our own implementation in `src/models/nafnet_sr.py`, trained from scratch on
the KLA data (no pretrained weights).

Final configuration (29,195,553 parameters):

```yaml
img_channels: 1
width: 32
enc_blocks: [2, 2, 4, 8]
middle_blocks: 12
dec_blocks: [2, 2, 2, 2]
scale: 2
```

**Training setup (final model, experiment `nafnet_sr_A_e80`):**

| | value |
|---|---|
| epochs | 80 (best checkpoint at epoch 55) |
| batch size | 16 |
| LR patch | 128 |
| optimizer | AdamW, initial LR 3e-4, cosine schedule |
| precision | mixed precision (AMP) |
| weight averaging | EMA (0.999) |
| loss | `1.0 × Charbonnier + 0.1 × SSIM` |
| seed | 42 |

Details in [`docs/final_model.md`](docs/final_model.md).

---

## Install

Requires Python 3.11. A GPU is recommended for training and fast inference; inference also
runs on CPU.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If your CUDA version differs, install a matching torch build, e.g. CUDA 11.8:
#   pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

The trained checkpoint is provided separately (it is large and not committed to git).
Place it at:

```
weights/nafnet_sr_A_e80/best.pt
```

---

## Run inference

Standalone — takes an input directory of degraded `.npy` files and writes restored `.npy`
files with the same names. No source edits needed.

```bash
python inference.py \
    --input_dir data/test/NoisyLR \
    --output_dir output \
    --checkpoint weights/nafnet_sr_A_e80/best.pt \
    --batch 16
```

Each output is `(256, 256)` float32, clipped to `[0, 1]`, saved as `<id>.npy`.

## Run the benchmark

Measures the complete pipeline (disk read → preprocess → transfer → model → postprocess →
save), not just the forward pass.

```bash
python benchmark.py --input_dir data/test/NoisyLR --checkpoint weights/nafnet_sr_A_e80/best.pt --batch 16
```

## Reproduce training

The dataset is provided as `train.zip` + `Test_NoisyLR.zip`. Stage it (this excludes macOS
junk and is idempotent):

```bash
python scripts/setup_data.py --src semicon_dataset   # folder containing the two zips
python scripts/validate_data.py                      # integrity check
```

The split (`configs/split.json`) ships with the repo — use it as-is. Then train the final
model (16 GB GPU; ~1.5–2 h on a T4):

```bash
python train.py --config configs/nafnet_sr_16gb.yaml --epochs 80 --exp-id nafnet_sr_A_e80
```

This writes per-metric checkpoints and a `history.json` under `weights/nafnet_sr_A_e80/`.
Training was done on a Google Colab T4 — see [`docs/kaggle_colab_setup.md`](docs/kaggle_colab_setup.md)
and [`docs/InsightFab_200epoch_Colab.ipynb`](docs/InsightFab_200epoch_Colab.ipynb) for the
cloud workflow (set `--epochs 80` for the final recipe).

---

## Results (held-out validation — NOT hidden-test scores)

These are measured on our own held-out validation split (`configs/split.json`), **not** on
KLA's hidden test set. The official competition score is computed separately by KLA on their
hardware and hidden ground truth.

| Method | IID PSNR | IID SSIM | IID LPIPS | OOD PSNR | OOD SSIM | OOD LPIPS |
|---|---|---|---|---|---|---|
| Bicubic ×2 (reference floor) | 22.86 | 0.537 | 0.452 | 21.47 | 0.390 | 0.561 |
| Small CNN (reference baseline) | 27.69 | 0.753 | 0.296 | 28.68 | 0.677 | 0.427 |
| **NAFNet-SR (final, epoch 55)** | **28.5625** | **0.78622** | **0.23497** | **29.6334** | **0.71502** | **0.2875** |

Metrics: PSNR ↑, SSIM ↑, LPIPS ↓. The OOD-val set is a held-out content cluster; its PSNR
can read higher than IID because those images are smoother, so we also watch SSIM/LPIPS for
the real generalization picture.

**Local runtime sanity check (NOT the official runtime):** 400 images restored on CPU in
224.01 s (≈ 1.8 images/sec). This is only a local sanity measurement; the official
end-to-end runtime is evaluated by KLA on the competition hardware (GPU), where it is much
faster.

---

## Repository structure

```
inference.py            # standalone inference (--input_dir / --output_dir)
benchmark.py            # end-to-end runtime benchmark
train.py                # config-driven training entry point
requirements.txt        # dependencies (requirements-cloud.txt for Colab/Kaggle)
configs/                # split.json, content_clusters.json, model/training configs
src/
  data/                 # paired dataset + split loading
  models/               # NAFNet-SR (+ small-CNN baseline) and a model registry
  losses/               # Charbonnier, SSIM (+ optional MS-SSIM / HF used in the B experiment)
  evaluation/           # PSNR / SSIM / LPIPS + validation loop
  inference/            # shared restorer + self-describing checkpoint format
  augmentation/         # degradation simulator used by the (rejected) B experiment
  training/             # trainer (AMP, EMA, resume, per-metric checkpoints)
  utils/                # npy I/O (range policy), paths, experiment logging
scripts/                # data setup, audit, split, baselines, comparison tools
tests/                  # unit/integration tests
docs/                   # dataset report, final-model notes, external resources, cloud setup
results/dataset_audit/  # measured dataset figures + statistics
experiments/            # baseline experiment records + index
weights/                # checkpoints (not committed; place best.pt here)
```

## Experiments summary

Two training approaches were compared on the same split:

- **A — baseline recipe** (real data, Charbonnier + 0.1·SSIM): this is the final model above.
- **B — degradation-aware** (real + on-the-fly synthetic re-degradation, MS-SSIM + high-frequency
  loss): scored worse on all six metrics and was **rejected**. Its tooling
  (`src/augmentation/`, `configs/nafnet_sr_degaug.yaml`, `scripts/verify_no_leakage.py`,
  `scripts/compare_ab.py`) is kept in the repo for reproducibility of that comparison.

External models/methods used (NAFNet architecture reference, LPIPS metric) are disclosed in
[`docs/external_resources.md`](docs/external_resources.md).

## Limitations

- The restoration is denoising-dominated; the clean images have little fine high-frequency
  detail, so gains are largely in noise removal and structure recovery rather than inventing
  texture.
- Validation numbers above are on our own split, not KLA's hidden test set; the two can differ.
- Any synthetic degradation in the (rejected) B experiment was generated only from the
  training GT images; validation/test images were never used to create training samples.
- The model does not intentionally sharpen or add structure that is not supported by the
  input, to avoid hallucinating detail that is not real.
