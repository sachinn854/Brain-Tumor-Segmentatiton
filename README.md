# Brain Tumor Segmentation — B.Tech Final Year Project

Multimodal MRI brain tumor segmentation on the BraTS dataset.
Approach: replicate an IEEE Transactions (2024+) base model exactly, then add and validate an original improvement.

**Timeline:** 10 Aug 2026 → 31 Dec 2026 · Guide meeting every Thursday

---

## Current status

**Base paper:** **WAS-Mamba** — *Windowed Attention State Space Model for 3D Medical Image Segmentation*, IEEE Transactions on Image Processing, vol. 35, 2026. Mamba/state-space architecture; evaluated across 5 datasets (Synapse, BTCV, ACDC, BraTS, Decathlon-Lung) — this project uses only its BraTS setup.

- Guide approval: pending
- Contribution idea: not finalized yet
- Base-model scaffold: done — architecture, loss, BraTS data loader, augmentation, and a full training loop. Model + loss + data loader verified end-to-end on a real BraTS case.
- Training runs **locally** (not Colab — the free T4's 16GB VRAM can't hold this 3D model's training activations even at batch size 1).

---

## Folder structure (this repo)

```
src/
├── models/wasmamba.py            WAS-Mamba architecture (copied from the authors'
│                                 repo + 4 bug fixes to make their own code run)
├── losses/losses.py              Loss functions, incl. the paper's Dice+CE loss
├── data/
│   ├── brats_dataset.py          BraTS PyTorch Dataset (written from scratch —
│   │                             the paper's public code has no BraTS loader)
│   └── augmentation.py           nnFormer's training augmentation pipeline
├── engine/train.py               Training loop — split, checkpointing, resume, validation
├── utils/                        seed, optimizer, scheduler, logging
└── configs/wasmamba_config.py    All hyperparameters, each annotated with its source
```

Planning notes, downloaded papers, and the literature review are kept locally only (not pushed here).

---

## Setup (local, Linux or WSL2)

Install PyTorch first, matched to your CUDA version (see pytorch.org). Then:

```bash
pip install -r requirements.txt
```

`mamba-ssm` / `causal-conv1d` compile CUDA extensions and need `nvcc` on PATH.
On Windows they don't install reliably — use WSL2 or a Linux machine.
If the isolated build can't see your torch: `pip install causal-conv1d mamba-ssm --no-build-isolation`

---

## Data

Put BraTS cases under `data/BraTS2021/` (gitignored), one folder per case:

```
data/BraTS2021/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00002/
│   └── ...
```

The official BraTS2021 training archive already has this layout.

---

## Train

```bash
python -m src.engine.train --epochs 2       # quick end-to-end check first
python -m src.engine.train                  # real run (1000 epochs, per the paper)
```

Config defaults (`batch_size=1`, gradient checkpointing on) are tuned for ~16GB VRAM.
On a ≥24GB GPU, match the paper's setup: `--batch_size 2 --no_checkpoint`.

Checkpoints go to `results/checkpoints/`. Training auto-resumes from `latest.pth` there,
so an interrupted run just needs the same command again.
