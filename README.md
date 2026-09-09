# Brain Tumor Segmentation — B.Tech Final Year Project

Multimodal MRI brain tumor segmentation on the BraTS dataset.
Approach: replicate an IEEE Transactions (2024+) base model exactly, then add and validate an original improvement.

**Timeline:** 10 Aug 2026 → 31 Dec 2026 · Guide meeting every Thursday

---

## 📄 Documents — is order me padho

| File | Kya hai |
|---|---|
| [`docs/00-MASTER-PLAN.md`](docs/00-MASTER-PLAN.md) | **Week-by-week plan** — August se December tak, 21 weeks |
| [`docs/01-PAPER-TYPES-EXPLAINED.md`](docs/01-PAPER-TYPES-EXPLAINED.md) | Journal vs **Transactions** vs Conference ka farak + IEEE Xplore pe search kaise karein |
| [`docs/02-SHORTLISTED-PAPERS.md`](docs/02-SHORTLISTED-PAPERS.md) | 5 shortlisted **IEEE Transactions papers (2024+)** + base model recommendation |
| [`docs/03-THURSDAY-MEETING-SCRIPT.md`](docs/03-THURSDAY-MEETING-SCRIPT.md) | 13 Aug guide meeting me **exactly kya bolna hai** |
| [`docs/07-TIP-NORMAL-BRAIN-BOOST-BREAKDOWN.md`](docs/07-TIP-NORMAL-BRAIN-BOOST-BREAKDOWN.md) | ⭐ **Option A** — Normal-Brain-Boost (IEEE TIP 2024). 2 modules, idea ek line ka, 2.5D input |
| [`docs/06-S2CA-Net-SIMPLE-VERSION.md`](docs/06-S2CA-Net-SIMPLE-VERSION.md) | ⭐ **Option B** — S²CA-Net (IEEE TMI 2024). 3 modules, par single pipeline aur single dataset |
| [`docs/04-S2CA-Net-PAPER-BREAKDOWN.md`](docs/04-S2CA-Net-PAPER-BREAKDOWN.md) | S²CA-Net ka **detailed** breakdown — Week 6 me code likhte waqt kholna, abhi nahi |
| [`docs/05-UNETR-PLUSPLUS-PAPER-BREAKDOWN.md`](docs/05-UNETR-PLUSPLUS-PAPER-BREAKDOWN.md) | UNETR++ ka breakdown — sirf 1 block, par general paper hai (purely brain tumor nahi). Backup |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Daily log — **roz 3 line likhna** |

---

## Current status

**Phase 0 — Literature Review** · Week 1 of 21

- Proposed base paper: **UNETR++** — *Delving Into Efficient and Accurate 3D Medical Image Segmentation*, IEEE Transactions on Medical Imaging, vol. 43, no. 9, pp. 3377–3390, 2024. Chuna kyunki isme sirf **ek naya block (EPA)** hai — ek semester me reproduce karna realistic hai
- Backup: **S²CA-Net**, IEEE TMI, vol. 43, no. 7, pp. 2495–2508, 2024 — purely brain tumor ka paper, par 3 naye modules hain
- Dataset: BraTS 2020 / 2021
- Status: guide ki approval pending (Thu 13 Aug)

---

## Phases

| Phase | Weeks | Output |
|---|---|---|
| 0 — Literature | 1–2 | Base paper locked + method samjha |
| 1 — Setup & Data | 3–5 | GPU env + BraTS pipeline + U-Net sanity run |
| 2 — Base Model | 6–9 | Paper ka model exactly reproduced |
| 3 — Validation | 10–12 | Numbers paper ke ±2-3% me + limitations documented |
| 4 — Contribution | 13–17 | Mera improvement + ablation study |
| 5 — Delivery | 18–20 | Report + PPT + demo |
| Buffer | 21 | Submission |

---

## Folder structure

```
├── docs/         planning aur notes
├── papers/       downloaded PDFs
├── data/         BraTS dataset (gitignored)
├── notebooks/    exploration
├── src/          models, data, losses, train.py
├── results/      tables, figures, logs
└── report/       final report
```
