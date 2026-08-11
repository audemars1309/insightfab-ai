# External Resources Disclosure

Per KLA rules, every external model/dataset is disclosed with name, link, licence,
reference, and how it is used. This file is updated whenever a new external resource
is added.

## Current status
- **No external training datasets** are used. Training uses only the official KLA
  paired data (and, optionally, synthetic pairs derived from the official GT).
- **No pretrained restoration weights** are used to initialize the model (from scratch).

## External models in use

### 1. LPIPS (AlexNet backbone) — evaluation metric only
| Field | Value |
|---|---|
| Name | LPIPS (Learned Perceptual Image Patch Similarity), `lpips` v0.1.4, `net='alex'` |
| Link | https://github.com/richzhang/PerceptualSimilarity |
| Paper | Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018 |
| Licence | BSD-2-Clause |
| Backbone weights | torchvision AlexNet, ImageNet-1k pretrained (`alexnet-owt-7be5be79.pth`), BSD-3-Clause (PyTorch) |
| **Used for** | Computing the **LPIPS metric** during validation/reporting only. |
| **NOT used for** | Training loss, model initialization, or the competition inference path. |

> LPIPS is one of the three official reported metrics, so it is a measurement tool here,
> not part of the restoration model. The competition `inference.py` does not import it.

### 2. NAFNet — architecture reference (reimplemented, no external weights)
| Field | Value |
|---|---|
| Name | NAFNet (Nonlinear Activation Free Network) |
| Link | https://github.com/megvii-research/NAFNet |
| Paper | Chen et al., "Simple Baselines for Image Restoration", ECCV 2022 |
| Licence | MIT |
| **Used for** | Architecture design of `src/models/nafnet_sr.py` (our own implementation, trained **from scratch** on KLA data). |
| **NOT used** | No NAFNet pretrained weights are used. |

### 3. VGG16 perceptual loss — OPTIONAL, off by default
| Field | Value |
|---|---|
| Name | torchvision VGG16, ImageNet-1k pretrained |
| Link | https://pytorch.org/vision/stable/models.html |
| Licence | BSD-3-Clause |
| **Used for** | Optional perceptual training term (`loss.w_perceptual`), **disabled (0.0) by default** to avoid hallucinating structure. Enable only with a small weight after verifying it does not degrade PSNR/SSIM or invent detail. |

## Software / frameworks
PyTorch, torchvision, scikit-image (PSNR/SSIM), scikit-learn (content clustering for the
OOD split), NumPy, SciPy, matplotlib — standard permissively-licensed libraries; exact
versions pinned in `requirements.txt`.

## KLA-referenced literature (background reading, not code dependencies)
- Zhai et al. (2023), *A Comprehensive Review of Deep Learning-Based Real-World Image Restoration*, IEEE Access 11.
- Terven et al. (2025), *A Comprehensive Survey of Loss Functions and Metrics in Deep Learning*, Artif. Intell. Rev. 58.
- Monga et al. (2021), *Algorithm Unrolling*, IEEE Signal Processing Magazine 38(2).
- Kumar et al. (2024), *Image Data Augmentation Approaches: A Comprehensive Survey*, IEEE Access 12.
