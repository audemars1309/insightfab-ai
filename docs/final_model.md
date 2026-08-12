# Final Model — `nafnet_sr_A_e80`

This is the model submitted for the hackathon.

## Architecture
NAFNet-SR (our implementation, `src/models/nafnet_sr.py`), trained from scratch on the KLA
training data. **29,195,553 parameters.**

```yaml
model:
  name: nafnet_sr
  kwargs:
    img_channels: 1
    width: 32
    enc_blocks: [2, 2, 4, 8]
    middle_blocks: 12
    dec_blocks: [2, 2, 2, 2]
    scale: 2
    drop: 0.0
```

## Training
| setting | value |
|---|---|
| experiment id | `nafnet_sr_A_e80` |
| epochs | 80 |
| selected checkpoint | epoch 55 (best on validation) |
| batch size | 16 |
| LR patch | 128 |
| optimizer | AdamW, initial LR 3e-4, cosine schedule to 1e-6 |
| precision | mixed precision (AMP) |
| weight averaging | EMA, decay 0.999 |
| loss | `1.0 × Charbonnier + 0.1 × SSIM` |
| seed | 42 |
| data | real KLA training pairs only |

Reproduce:
```bash
python train.py --config configs/nafnet_sr_16gb.yaml --epochs 80 --exp-id nafnet_sr_A_e80
```
(Config `configs/nafnet_sr_16gb.yaml` defines the architecture, loss weights, optimizer, AMP
and EMA; `--epochs 80` sets the 80-epoch schedule.)

## Checkpoint
`weights/nafnet_sr_A_e80/best.pt` — self-describing (stores the model name + kwargs), so
`inference.py` loads it without any code changes.

## Validation results (held-out validation split — NOT hidden-test / leaderboard scores)
Measured on `configs/split.json` (IID-val 293, OOD-val 256) with the epoch-55 checkpoint:

| split | PSNR | SSIM | LPIPS |
|---|---|---|---|
| IID | 28.5625 | 0.78622 | 0.23497 |
| OOD | 29.6334 | 0.71502 | 0.28750 |

These are our own validation numbers. The official competition score is computed by KLA on
their hidden test set and hardware, and is not represented here.

## Why this recipe
A controlled A/B experiment compared this baseline recipe (A) against a degradation-aware
variant (B: real + synthetic re-degradation, MS-SSIM + high-frequency loss). B was worse on
all six metrics and was rejected. Two independent runs of recipe A (an earlier 200-epoch run
and this 80-epoch run) converged to essentially the same validation quality
(~28.6 IID PSNR / ~0.786 SSIM), i.e. a stable ceiling for this model and data.
