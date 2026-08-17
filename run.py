"""KLA image-restoration inference — official evaluator entry point.

Usage (exactly as the evaluator runs it):

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it with the NAFNet-SR A-80 checkpoint
bundled at models/best.pt, and writes one restored .npy per input to
<output-dir> under the SAME filename.

Output contract (enforced here):
    * grayscale (H, W)
    * 256 x 256
    * float32
    * values clipped to [0, 1]
    * no NaN, no Inf

Fully offline: no network calls, no downloads, no API keys, no LPIPS/torchvision.
The model architecture below is copied verbatim from the training repository
(src/models/nafnet_sr.py); nothing about the model is changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Model — NAFNet-SR (verbatim from src/models/nafnet_sr.py). Do not modify.    #
# --------------------------------------------------------------------------- #


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over (N, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)  # depthwise
        self.conv3 = nn.Conv2d(dw // 2, c, 1)                    # after SimpleGate halves ch
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.sg = SimpleGate()

        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.drop2(x) * self.gamma


class NAFNetSR(nn.Module):
    def __init__(
        self,
        img_channels: int = 1,
        width: int = 32,
        enc_blocks=(2, 2, 4, 8),
        middle_blocks: int = 12,
        dec_blocks=(2, 2, 2, 2),
        scale: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        self.scale = scale
        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(middle_blocks)])

        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(n)]))

        # SR tail: LR features (width ch) -> HR (img_channels), no checkerboard.
        self.sr_tail = nn.Sequential(
            nn.Conv2d(width, width * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, img_channels, 3, padding=1),
        )
        self.padder_size = 2 ** len(enc_blocks)

    def _check_pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, h, w

    def forward(self, inp):
        base = F.interpolate(inp, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x, H, W = self._check_pad(inp)

        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        out = self.sr_tail(feat)                       # HR residual, (N,1,scale*Hpad,scale*Wpad)
        out = out[:, :, : H * self.scale, : W * self.scale]
        return base + out


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #

TARGET_SIZE = 256                      # required output resolution (H = W = 256)
CHECKPOINT = Path(__file__).resolve().parent / "models" / "best.pt"


def load_model(device: str) -> NAFNetSR:
    """Load the bundled NAFNet-SR A-80 checkpoint. Never loads any other file."""
    ckpt = torch.load(CHECKPOINT, map_location=device)
    if ckpt.get("model_name") != "nafnet_sr":
        raise SystemExit(
            f"expected a nafnet_sr checkpoint, got model_name={ckpt.get('model_name')!r}"
        )
    model = NAFNetSR(**ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def load_input(path: Path) -> np.ndarray:
    """Load a degraded .npy as a 2-D float32 array. Inputs are NOT clipped."""
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:      # (H, W, 1) -> (H, W)
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise SystemExit(f"{path.name}: expected grayscale (H,W) or (H,W,1), got {arr.shape}")
    return arr


@torch.inference_mode()
def restore(model: NAFNetSR, arr: np.ndarray, device: str, fp16: bool) -> np.ndarray:
    """Restore one image -> (256, 256) float32 in [0, 1], finite."""
    t = torch.from_numpy(arr[None, None]).to(device)          # (1, 1, H, W)
    with torch.autocast(device_type="cuda", enabled=fp16):
        out = model(t)
    # Guarantee the required 256x256 output regardless of input size.
    if out.shape[-2:] != (TARGET_SIZE, TARGET_SIZE):
        out = F.interpolate(
            out.float(), size=(TARGET_SIZE, TARGET_SIZE), mode="bicubic", align_corners=False
        )
    out = out[0, 0].float().cpu().numpy()
    # Scrub non-finite values, then clip to the target [0, 1] range.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: python run.py <input-dir> <output-dir>")
    input_dir, output_dir = Path(argv[0]), Path(argv[1])

    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")
    if not CHECKPOINT.exists():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy files found in {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    model = load_model(device)
    print(f"device={device}  checkpoint={CHECKPOINT.name}  images={len(files)}")

    for path in files:
        restored = restore(model, load_input(path), device, fp16)
        np.save(output_dir / path.name, restored)

    print(f"restored {len(files)} images -> {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
