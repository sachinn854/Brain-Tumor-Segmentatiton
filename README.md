# Brain Tumor Segmentation — B.Tech Final Year Project

Multimodal MRI brain tumor segmentation on the BraTS dataset.
Approach: replicate an IEEE Transactions (2024+) base model exactly, then add and validate an original improvement.

**Timeline:** 10 Aug 2026 → 31 Dec 2026 · Guide meeting every Thursday

---

## Current status

**Base paper:** **WAS-Mamba** — *Windowed Attention State Space Model for 3D Medical Image Segmentation*, IEEE Transactions on Image Processing, vol. 35, 2026. Mamba/state-space architecture; evaluated across 5 datasets (Synapse, BTCV, ACDC, BraTS, Decathlon-Lung) — this project uses only its BraTS setup.

- Guide approval: pending
- Contribution idea: not finalized yet
- Base-model code scaffold: done (`src/`) — architecture, loss, BraTS data loader, and hyperparameters cross-checked against the paper (see file-level comments in `src/configs/wasmamba_config.py` for exactly what's paper-confirmed vs. inferred)
- Not yet done: real training run (blocked on GPU access beyond local dev/debug — see that config file's comments), data augmentation pipeline

---

## Folder structure (this repo)

```
src/
├── models/wasmamba.py       WAS-Mamba architecture
├── losses/losses.py         Loss functions, incl. the paper's Dice+CE loss
├── data/brats_dataset.py    BraTS PyTorch Dataset (written from scratch —
│                             not part of the paper's public code release)
├── utils/                   Training utilities (seed, optimizer, scheduler,
│                             logging) and evaluation metrics
└── configs/wasmamba_config.py  Training config, hyperparameters
```

Planning notes, downloaded papers, and the literature review are kept locally only (not pushed here).

---

## Setup

```bash
pip install torch einops timm mamba-ssm causal-conv1d
```

`mamba-ssm`'s CUDA kernels are Linux-targeted; on Windows, use WSL2 or run on Colab/a Linux GPU box.
