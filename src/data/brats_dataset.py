"""
BraTS PyTorch Dataset for WAS-Mamba — written from scratch this session.

WHY THIS FILE EXISTS: the WAS-Mamba GitHub repo never released a BraTS
data-loading script — the only dataset code it ships (`config_setting.py`'s
`from datasets.dataset import *`) is for Synapse, and even that import
target doesn't exist in the repo. So there is nothing to "port" here; this
is a fresh implementation aimed at matching what the paper (Section IV-A.4)
says about BraTS, not a translation of released code.

Expects the standard per-case BraTS folder layout, e.g. (BraTS2020/2021 style):
    <root>/<case_id>/<case_id>_flair.nii.gz
    <root>/<case_id>/<case_id>_t1.nii.gz
    <root>/<case_id>/<case_id>_t1ce.nii.gz   (= paper's "T1Gd")
    <root>/<case_id>/<case_id>_t2.nii.gz
    <root>/<case_id>/<case_id>_seg.nii.gz
If your downloaded BraTS version names files differently, adjust MODALITY_SUFFIXES.

Design choices below that the PAPER DOES NOT SPECIFY (flagged, not hidden):
  - Per-modality z-score normalization over brain (nonzero) voxels only —
    this is the near-universal convention in BraTS code (nnU-Net, nnFormer,
    UNETR++ all do this), so a reasonable default, but not confirmed
    against this specific paper's text.
  - Random crop to 128^3 for training, center crop for val/test — the
    paper states the *trained* resolution (128x128x128) but not the crop
    strategy. Same caveat.
  - Raw BraTS labels are {0, 1, 2, 4} (label 3 is unused; 4 = enhancing
    tumor). These are remapped to {0, 1, 2, 3} below so they're valid
    class indices for CrossEntropyLoss / nDiceLoss with num_classes=4.
    This remapping is standard, not paper-specific.

Needs: pip install nibabel
"""

import os
import glob

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

MODALITY_SUFFIXES = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
SEG_SUFFIX = '_seg.nii.gz'

# raw BraTS label -> contiguous class index (paper's 3 targets + background)
# 0 background, 1 non-enhancing tumor core, 2 edema, 4 enhancing tumor
RAW_LABEL_TO_CLASS = {0: 0, 1: 1, 2: 2, 4: 3}


def _zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize using only nonzero (brain) voxels, like nnU-Net/nnFormer do."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std < 1e-8:
        std = 1e-8
    out = (volume - mean) / std
    out[~mask] = 0
    return out


def _remap_labels(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.int64)
    for raw, cls in RAW_LABEL_TO_CLASS.items():
        out[seg == raw] = cls
    return out


def _center_crop_or_pad(volume: np.ndarray, target_shape):
    """volume: (C, H, W, D) or (H, W, D). Crops/pads the last 3 dims to target_shape."""
    is_multichannel = volume.ndim == 4
    spatial = volume.shape[1:] if is_multichannel else volume.shape

    pad_width = []
    if is_multichannel:
        pad_width.append((0, 0))
    starts = []
    for dim_size, target in zip(spatial, target_shape):
        if dim_size < target:
            pad_before = (target - dim_size) // 2
            pad_after = target - dim_size - pad_before
            pad_width.append((pad_before, pad_after))
            starts.append(0)
        else:
            pad_width.append((0, 0))
            starts.append((dim_size - target) // 2)

    volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
    if is_multichannel:
        volume = volume[
            :,
            starts[0]:starts[0] + target_shape[0],
            starts[1]:starts[1] + target_shape[1],
            starts[2]:starts[2] + target_shape[2],
        ]
    else:
        volume = volume[
            starts[0]:starts[0] + target_shape[0],
            starts[1]:starts[1] + target_shape[1],
            starts[2]:starts[2] + target_shape[2],
        ]
    return volume


def _random_crop(image: np.ndarray, label: np.ndarray, target_shape):
    """image: (C, H, W, D), label: (H, W, D). Same crop location for both."""
    _, H, W, D = image.shape
    th, tw, td = target_shape
    h0 = np.random.randint(0, max(H - th, 0) + 1)
    w0 = np.random.randint(0, max(W - tw, 0) + 1)
    d0 = np.random.randint(0, max(D - td, 0) + 1)
    image = image[:, h0:h0 + th, w0:w0 + tw, d0:d0 + td]
    label = label[h0:h0 + th, w0:w0 + tw, d0:d0 + td]
    return image, label


class BratsDataset(Dataset):
    """
    Args:
        base_dir: folder containing one subfolder per case
        split: 'train', 'val', or 'test' — only affects crop strategy
               (random vs. center); the actual 80:5:15 case split should be
               decided by which case IDs you pass in `case_ids`
        case_ids: explicit list of case folder names for this split.
                  Splitting is left to the caller (e.g. a fixed 80:5:15
                  random split with a saved seed) rather than baked in here,
                  so exactly which cases land in train/val/test is
                  reproducible and inspectable.
        crop_size: (H, W, D), defaults to the paper's 128x128x128
    """

    def __init__(self, base_dir, split='train', case_ids=None, crop_size=(128, 128, 128)):
        self.base_dir = base_dir
        self.split = split
        self.crop_size = crop_size
        if case_ids is not None:
            self.case_ids = list(case_ids)
        else:
            self.case_ids = sorted(
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            )

    def __len__(self):
        return len(self.case_ids)

    def _load_case(self, case_id):
        case_dir = os.path.join(self.base_dir, case_id)
        modality_volumes = []
        for suffix in MODALITY_SUFFIXES:
            matches = glob.glob(os.path.join(case_dir, f'*{suffix}'))
            if not matches:
                raise FileNotFoundError(f"Missing modality {suffix} for case {case_id} in {case_dir}")
            vol = nib.load(matches[0]).get_fdata().astype(np.float32)
            modality_volumes.append(_zscore_normalize(vol))
        image = np.stack(modality_volumes, axis=0)  # (4, H, W, D)

        seg_matches = glob.glob(os.path.join(case_dir, f'*{SEG_SUFFIX}'))
        if not seg_matches:
            raise FileNotFoundError(f"Missing segmentation mask for case {case_id} in {case_dir}")
        label = nib.load(seg_matches[0]).get_fdata().astype(np.int64)
        label = _remap_labels(label)

        return image, label

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        image, label = self._load_case(case_id)

        if self.split == 'train':
            image, label = _random_crop(image, label, self.crop_size)
        else:
            image = _center_crop_or_pad(image, self.crop_size)
            label = _center_crop_or_pad(label, self.crop_size)

        image = torch.from_numpy(image).float()          # (4, H, W, D)
        label = torch.from_numpy(label).long()            # (H, W, D)
        return {'image': image, 'label': label, 'case_id': case_id}
