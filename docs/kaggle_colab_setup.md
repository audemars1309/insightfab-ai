# Cloud Training — NAFNet-SR, 200 epochs, resumable (Kaggle / Colab)

The competition model (`configs/nafnet_sr_16gb.yaml`, ~29.2M params) trains on a single
**16 GB GPU**. Local dev used a 4 GB GTX 1650 (smoke tests only); the code is identical.
This guide is the **manual handoff** — Claude Code cannot drive the cloud GPU, so you run
these steps and bring the outputs back.

**Fixed run identity (important):** always pass the SAME `--exp-id` on the first run and on
every resume, so all checkpoints land in one folder and it stays ONE experiment. This guide
uses `--exp-id nafnet_sr_16gb_e200`.

> Do **not** retrain on the hidden test inputs. Training uses only `data/train/`.

---

## What you must NOT change (already validated)
Dataset, split (`configs/split.json`, seed 42), IID/OOD methodology, baselines,
`inference.py`, `benchmark.py`, the NAFNet-SR architecture, experiment tracking, the
synthetic simulator. Just run the commands below.

---

## Option A — Kaggle (recommended: free 16 GB T4/P100, ~30 h/week)

### One-time setup
1. Push this repo to GitHub.
2. Create a **private Kaggle Dataset** containing `train.zip` and `Test_NoisyLR.zip`
   (say it mounts at `/kaggle/input/semicon-dataset/`).
3. New **Notebook** → Settings: **Accelerator = GPU T4 x2 or P100**, **Internet = On**,
   **Persistence = Variables and Files** (so `/kaggle/working` survives between sessions).
   > Use only **one** GPU (the config is single-GPU). T4 x2 is fine; training uses GPU 0.

### The notebook (paste as cells)

**Cell 1 — setup (idempotent; safe to re-run every session):**
```bash
[ -d insightfab-ai ] || git clone https://github.com/<you>/insightfab-ai.git
cd insightfab-ai
pip install -q -r requirements-cloud.txt          # uses Kaggle's CUDA-matched torch
python scripts/setup_data.py --src /kaggle/input/semicon-dataset
python scripts/validate_data.py                   # integrity check only
```

> **Use the committed split — do NOT regenerate it on the cloud.** `configs/split.json` and
> `configs/content_clusters.json` are in the repo and are the validated IID/OOD split. Do
> **not** run `build_split.py`/`audit_dataset.py` before training: a different sklearn/BLAS
> build could produce different K-means clusters and thus a different split. (Regenerating
> the audit *figures* is fine later, but training must use the committed `split.json`.)

**Cell 2 — train / auto-resume (same cell handles both):**
```bash
cd insightfab-ai
EXP=nafnet_sr_16gb_e200
RESUME=weights/$EXP/resume.pt
if [ -f "$RESUME" ]; then
  echo "Resuming $EXP from $RESUME"
  python train.py --config configs/nafnet_sr_16gb.yaml --exp-id $EXP --resume $RESUME
else
  echo "Starting fresh: $EXP"
  python train.py --config configs/nafnet_sr_16gb.yaml --exp-id $EXP
fi
```
With **Persistence = Files** on, `weights/$EXP/resume.pt` from the previous session is still
there, so re-running Cell 2 continues at the next epoch (never restarts at 1). The run
checkpoints **every epoch** (best/last/resume), so any number of sessions works.

> Rough time: ~1–3 min/epoch on a T4 ⇒ 200 epochs ≈ 4–10 h, i.e. 1–3 Kaggle sessions.
> If a 12 h session ends mid-run, just open the notebook again and re-run Cells 1–2.

### For a long unattended run
Use **Save Version → "Save & Run All (Commit)"** — it executes headless up to 12 h and
persists `/kaggle/working` as the version Output. Commit again to continue (Persistence keeps
`weights/`), or add the previous version's Output as input and copy `resume.pt` back first.

### After training — verify + benchmark (new cell)
```bash
cd insightfab-ai
EXP=nafnet_sr_16gb_e200
cp weights/$EXP/best.pt weights/final.pt

# Inference contract check (dirs only, no code edits)
python inference.py --input_dir data/test/NoisyLR --output_dir results/test_out --checkpoint weights/final.pt

# Full end-to-end benchmark ON THE CLOUD GPU (this must run on the cloud GPU)
python benchmark.py --input_dir data/test/NoisyLR --checkpoint weights/final.pt --batch 16 | tee results/benchmark_cloud.txt
```

> The **visual comparison** (`scripts/make_comparison.py`) is run **locally** after you bring
> `best.pt` back — it's inference-only (fits the 4 GB card) and the SmallCNN baseline
> checkpoint already lives in the local repo. No need to run it on the cloud.

---

## Option B — Google Colab (checkpoint to Drive for easy resume)

```python
# Runtime > Change runtime type > T4/L4 GPU
!git clone https://github.com/<you>/insightfab-ai.git
%cd insightfab-ai
!pip install -q -r requirements-cloud.txt
from google.colab import drive; drive.mount('/content/drive')
!python scripts/setup_data.py \
    --train-zip "/content/drive/MyDrive/semicon/train.zip" \
    --test-zip  "/content/drive/MyDrive/semicon/Test_NoisyLR.zip"
!python scripts/audit_dataset.py && python scripts/build_split.py --seed 42 && python scripts/validate_data.py

# Persist weights to Drive so resume survives disconnects:
!mkdir -p /content/drive/MyDrive/insightfab_weights && ln -sfn /content/drive/MyDrive/insightfab_weights weights
EXP="nafnet_sr_16gb_e200"
import os
resume = f"weights/{EXP}/resume.pt"
cmd = f"python train.py --config configs/nafnet_sr_16gb.yaml --exp-id {EXP}"
if os.path.exists(resume): cmd += f" --resume {resume}"
!{cmd}
```
Colab disconnects often; the Drive symlink means `resume.pt` persists — just re-run the cell.

---

## ⬇️ Files to download / bring back for evaluation

From `weights/nafnet_sr_16gb_e200/`:
- **`best.pt`** — best-validation checkpoint (this becomes `weights/final.pt`)
- **`last.pt`**, **`resume.pt`** — last epoch + resume state

From `experiments/`:
- **`nafnet_sr_16gb_e200.json`** (full record: IID+OOD PSNR/SSIM/LPIPS, params, hyperparams,
  train time) and the updated **`index.csv`**

From `results/`:
- **`benchmark_cloud.txt`** — the end-to-end benchmark from the cloud GPU
- **`comparisons/`** — the visual panels + `comparison_metrics.json`

Plus (paste as text): the **full training console log** (per-epoch loss + validation curve),
and the **GPU name + VRAM** (`!nvidia-smi`) and total wall-clock.

With these, Claude will: verify `best.pt` through `inference.py`, re-evaluate metrics, update
the README/report tables and the comparison figure, and write the evidence-based decision
(continue to more epochs / adjust loss / try synthetic / etc.). **No numbers are typed by
hand** — all come from these files.

---

## Memory / speed knobs (16 GB) — only if needed
| Situation | Change (CLI override, no file edits) |
|---|---|
| OOM | `--batch 8` (or `--lr-patch 96`) |
| GPU underused | `--batch 24` or `--batch 32` |
| — keep everything else at the config defaults — | |
