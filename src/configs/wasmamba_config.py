"""
Training config for WAS-Mamba on BraTS — base model, no contribution added yet.

Adapted from config_setting.py, https://github.com/1605066114/WAS-Mamba
(the original file hardwires the Synapse dataset and imports a `datasets`
package that was never released — see the long comment this replaced,
still true, kept short here).

Every number below marked "(paper)" comes from the WAS-Mamba paper itself
(IEEE TIP vol. 35, 2026), Section IV-A "Dataset" (4. BraTS for Brain Tumor
Segmentation) and Section IV-B "Implementation Details" — read in full
2026-09-08.

WAS-Mamba defers everything else to nnFormer and UNETR++:
    "All other training hyperparameters and data augmentation settings
     followed those of nnFormer [14] and UNETR++ [45]."
UNETR++ in turn defers to nnFormer for the same things ("All other
training hyper-parameters are same as in [nnFormer]") — so nnFormer is
the actual source for batch_size/scheduler/augmentation below, read
2026-09-08 (arXiv:2109.03201v6, Table I / Implementation Details).
Marked "(nnFormer, via WAS-Mamba's deferral)" for anything sourced that way.

One thing NOT carried over from nnFormer: it trains with deep supervision
(loss computed at 3 decoder resolutions). WAS-Mamba's own forward() (see
src/models/wasmamba.py) returns a single output tensor, not 3 — the
architecture has no deep-supervision heads, so this was deliberately not
force-fitted in. Trained single-output, matching what the code actually is.
"""

from datetime import datetime

from src.losses.losses import PaperDiceCeLoss
from src.data.brats_dataset import BratsDataset


class setting_config:
    """The config of training setting — WAS-Mamba, BraTS, base model."""

    network = 'WASMamba'
    model_config = {
        # (paper) BraTS: 4 input channels = FLAIR, T1w, T1Gd, T2w
        'input_channels': 4,
        # (paper) "targets divided into three classes: edema, enhancing
        # tumor, non-enhancing tumor" -> 3 foreground classes + background
        # = 4. NOT stated explicitly as "4" in the paper; this is our
        # inference from their class list. Confirm against Table V's
        # WT/TC/ET reporting before training (those are derived/composite
        # regions computed from these 3 raw classes, not the raw labels).
        'num_classes': 4,
        'depths': [1, 1, 1, 1],
        'depths_decoder': [1, 1, 1, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': None,
        # Gradient checkpointing -- ON because the paper's 48GB A6000 setup
        # OOMs on a 16GB Colab T4. Recomputes activations in backward
        # instead of storing them. Set False if training on >=24GB VRAM.
        'use_checkpoint': True,
    }

    datasets_name = 'brats'

    # (paper) "For ACDC, BraTS, and Decathlon-Lung, the models were trained
    # at resolutions of 128x128x16, 128x128x128, and 192x192x32" -> BraTS = 128^3
    input_size_h = 128
    input_size_w = 128
    input_size_c = 128
    patch_size = (2, 2, 2)  # (paper) "remaining datasets all use a 2x2x2 configuration"

    # No BraTS Dataset class existed anywhere in the original repo — this is
    # src/data/brats_dataset.py, written from scratch this session.
    # TODO: point data_path at wherever you download BraTS to, and decide
    # the actual 80:5:15 case-ID split (see BratsDataset's docstring —
    # splitting is deliberately left explicit/reproducible, not automatic).
    data_path = None       # e.g. './data/BraTS2021/train'
    datasets = BratsDataset
    list_dir = None
    volume_path = None     # e.g. './data/BraTS2021/val'

    # (paper) 80:5:15 train:val:test split
    train_split = 0.80
    val_split = 0.05
    test_split = 0.15

    # (nnFormer, via WAS-Mamba's deferral) training augmentation --
    # rotation, scaling, gaussian noise, gaussian blur, brightness/contrast,
    # low-resolution simulation, gamma, and mirroring, applied in that
    # order. Implemented 2026-09-09 in src/data/augmentation.py
    # (BraTSAugmentor), wired into BratsDataset for split='train' only.
    # Per-transform probabilities are nnU-Net's public defaults, not
    # independently confirmed against nnFormer's own code -- see that
    # file's docstring.

    pretrained_path = ''
    num_classes = model_config['num_classes']
    input_channels = model_config['input_channels']

    # (paper, Eq. 13 / Section III-E) Dice + cross-entropy — NOT the repo's
    # own CeDiceLoss (that one's CE term is dead/commented-out code, see
    # src/losses/losses.py for the mismatch note).
    criterion = PaperDiceCeLoss(num_classes)

    z_spacing = 1

    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False

    # Paper / nnFormer use batch_size = 2 (on a 48GB A6000). Dropped to 1
    # for the 16GB Colab T4 -- batch=2 OOMs even with gradient checkpointing
    # on. This is a hardware-forced deviation, worth stating in the report;
    # raise back to 2 if training on >=24GB VRAM.
    batch_size = 1

    # (paper) "WAS-Mamba was trained for 1k epochs on a single NVIDIA RTX
    # A6000 GPU with 48GB of memory"
    epochs = 1000

    work_dir = 'results/' + network + '_' + datasets_name + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    print_interval = 20
    val_interval = 1
    test_weights_path = ''

    threshold = 0.5

    # (paper, Section IV-B "Implementation Details") — note beta1=0.7 is
    # NOT the PyTorch default (0.9); this is deliberate per the paper, keep it.
    opt = 'Adam'
    lr = 0.01
    betas = (0.7, 0.999)
    eps = 1e-8
    weight_decay = 3e-5
    amsgrad = False

    # (nnFormer, via WAS-Mamba's deferral) polynomial decay:
    # lr = initial_lr * (1 - epoch/max_epoch)^0.9 — NOT StepLR (that was
    # this project's earlier guess, before nnFormer/UNETR++ were read).
    sch = 'PolyLR'
    poly_power = 0.9
