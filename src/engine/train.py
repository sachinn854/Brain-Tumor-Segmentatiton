"""
Training loop for WAS-Mamba base model on BraTS -- local training.

Written from scratch this session -- the original repo's own training
script (`test.py` in their GitHub release) imports `datasets.dataset`,
`engine_synapse`, and `configs.config_setting_synapse`, none of which exist
in the repo, so it never actually ran even for the authors' own Synapse
setup. There was nothing usable to port for BraTS specifically.

Data layout expected under --data_path (default: ./data/BraTS2021):
    <data_path>/<case_id>/<case_id>_flair.nii.gz
    <data_path>/<case_id>/<case_id>_t1.nii.gz
    <data_path>/<case_id>/<case_id>_t1ce.nii.gz
    <data_path>/<case_id>/<case_id>_t2.nii.gz
    <data_path>/<case_id>/<case_id>_seg.nii.gz
i.e. one folder per case, each holding that case's 4 modalities + mask.
The official BraTS2021 training archive already has this structure; if
yours is flat, wrap each case's files in a `<case_id>/` folder first.

Usage (run from the repo root):
    python -m src.engine.train                          # uses all defaults
    python -m src.engine.train --epochs 2               # quick end-to-end check first
    python -m src.engine.train --batch_size 2 --no_checkpoint   # if you have >=24GB VRAM

Resumes automatically from `latest.pth` in --checkpoint_dir if it exists,
so an interrupted run just needs the same command again to continue.

Defaults (batch_size=1, gradient checkpointing on) come from
src/configs/wasmamba_config.py and are tuned for a ~16GB GPU. On a bigger
card, pass --batch_size 2 --no_checkpoint to match the paper's setup.

Logging: a tqdm progress bar shows live loss + Dice score per step (not
just dice_loss=1-Dice, the actual overlap metric too). Every
config.print_interval steps and at the end of every epoch, both train and
val report is broken into CE and Dice loss components separately (not just
the combined loss actually used for backward) and Dice per class by name
(background is never reported, only the 3 foreground classes -- see
CLASS_NAMES, matching the label remapping in brats_dataset.py). HD95 (95th
percentile Hausdorff Distance, in voxels) is also reported per class, but
only at validation, once per epoch -- not every training step, since
distance transforms per class would meaningfully slow down all 1000
training steps/epoch for a metric that's only really meaningful at
val/test time anyway (matches how Table V in the paper reports it).
Everything printed also goes to <checkpoint_dir>/log/train.info.log.
"""

import os

# Reduce CUDA memory fragmentation -- must be set before torch is imported.
# On a 16GB T4 the difference between fitting and an OOM can be just the
# fragmented-but-unallocated slack this reclaims.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion

from src.models.wasmamba import WASMamba
from src.data.brats_dataset import BratsDataset
from src.utils.train_utils import set_seed, get_optimizer, get_scheduler, get_logger, log_config_info
from src.configs.wasmamba_config import setting_config as config

# Matches RAW_LABEL_TO_CLASS in src/data/brats_dataset.py (0=background is
# never reported per-class below, only the 3 foreground classes are).
CLASS_NAMES = ['background', 'non_enhancing_tumor_core', 'edema', 'enhancing_tumor']


def _format_per_class(values):
    """[0.12, 0.5, 0.8] -> 'non_enhancing_tumor_core: 0.1200, edema: 0.5000, enhancing_tumor: 0.8000'
    NaN (e.g. a class with no ground-truth voxels this epoch, see
    _hd95_score) prints as 'n/a' rather than a misleading number."""
    return ', '.join(
        f'{CLASS_NAMES[c + 1]}: {"n/a" if np.isnan(v) else f"{v:.4f}"}'
        for c, v in enumerate(values)
    )


def _dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Binary Dice for one class. Pure numpy -- deliberately NOT imported from
    src/utils/metrics.py, which pulls in SimpleITK/matplotlib/medpy at
    module load for its Synapse-specific 2D helpers (none of which are
    installed on a stock Colab, and none of which training needs).

    Convention when a class is absent from the ground truth: if the model
    also predicted nothing for it, that's a perfect 1.0; if it predicted
    something, that's 0.0.
    """
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    if gt_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def _hd95_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    95th-percentile Hausdorff Distance for one class, in voxels. Pure
    scipy (distance_transform_edt + binary_erosion) -- same reasoning as
    _dice_score above, no medpy/SimpleITK dependency needed.

    Only meaningful (and only computed) at VALIDATION time, not every
    training step -- distance transforms are cheap for one volume but not
    cheap enough to add to every one of 1000 training steps/epoch without
    undoing the num_workers speed fix. See validate() below.

    Returns NaN when the ground truth has no voxels of this class (HD95
    is undefined without a GT surface to measure against) -- callers
    should use np.nanmean when averaging across cases/classes so those
    don't silently count as 0.
    """
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    if gt_sum == 0:
        return 0.0 if pred_sum == 0 else float('nan')
    if pred_sum == 0:
        return float('nan')

    pred_border = pred_mask ^ binary_erosion(pred_mask)
    gt_border = gt_mask ^ binary_erosion(gt_mask)

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    dists_pred_to_gt = dt_gt[pred_border]
    dists_gt_to_pred = dt_pred[gt_border]
    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])

    if all_dists.size == 0:
        return 0.0
    return float(np.percentile(all_dists, 95))


def make_splits(data_path, train_split, val_split, seed, splits_path, max_train_cases=None):
    """
    Reproducible 80:5:15 (paper's ratios) train/val/test case-ID split.

    Written once to `splits_path` and reused on every later call/resume, so
    which cases land in which split is fixed and inspectable -- not
    re-randomized on every run (which would silently leak "test" cases into
    training across restarts).

    max_train_cases: if set, truncates ONLY the train list to this many
    cases (val/test keep their full, paper-ratio sizes -- so evaluation
    stays representative of the whole dataset even when training is capped
    for speed). This is a compute-budget compromise, not part of the paper;
    state the actual number used in the report.
    """
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            return json.load(f)

    case_ids = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_ids = case_ids[:n_train]
    if max_train_cases is not None:
        train_ids = train_ids[:max_train_cases]

    splits = {
        'train': train_ids,
        'val': case_ids[n_train:n_train + n_val],
        'test': case_ids[n_train + n_val:],
    }
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    return splits


def validate(model, val_loader, criterion, num_classes, device, verbose=True):
    """
    Per-class Dice + HD95 + CE/Dice loss breakdown on the val split. Unlike
    src/utils/metrics.py's `test_single_volume` (which is Synapse's
    2D-slice-based evaluation), this operates directly on the 3D volumes
    BratsDataset returns -- written fresh for BraTS rather than reusing
    that function.

    HD95 is computed here (not in the training loop) deliberately --
    distance transforms per class per case are cheap once per epoch over
    ~5% of cases, but would meaningfully slow down every one of 1000
    training steps/epoch if computed there too.
    """
    model.eval()
    total_loss, total_ce, total_dice_loss = 0.0, 0.0, 0.0
    dice_per_class = [[] for _ in range(1, num_classes)]
    hd95_per_class = [[] for _ in range(1, num_classes)]

    iterator = tqdm(val_loader, desc='  validating', leave=False, disable=not verbose)
    with torch.no_grad():
        for batch in iterator:
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            output = model(image)
            loss = criterion(output, label)
            total_loss += loss.item()
            # criterion (PaperDiceCeLoss) stores its two components as
            # sub-modules -- reused here for the breakdown, not re-derived.
            total_ce += criterion.celoss(output, label.long()).item()
            total_dice_loss += criterion.diceloss(output, label, softmax=True).item()

            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()
            gt = label.cpu().numpy()
            for c in range(1, num_classes):
                pred_c, gt_c = (pred == c), (gt == c)
                dice_per_class[c - 1].append(_dice_score(pred_c, gt_c))
                hd95_per_class[c - 1].append(_hd95_score(pred_c, gt_c))

    n = max(len(val_loader), 1)
    mean_loss = total_loss / n
    mean_ce = total_ce / n
    mean_dice_loss = total_dice_loss / n
    mean_dice_per_class = [float(np.mean(d)) if d else 0.0 for d in dice_per_class]
    # nanmean: a case where this class had no GT voxels contributes NaN
    # (see _hd95_score docstring) and should be excluded, not counted as 0.
    mean_hd95_per_class = [float(np.nanmean(h)) if h else float('nan') for h in hd95_per_class]
    return mean_loss, mean_ce, mean_dice_loss, mean_dice_per_class, mean_hd95_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/BraTS2021',
                         help='Folder containing one subfolder per BraTS case (default: data/BraTS2021)')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                         help='Where to save/resume checkpoints (default: results/checkpoints)')
    parser.add_argument('--splits_path', type=str, default=None,
                         help='Where to save/load the train/val/test case-ID split (default: <checkpoint_dir>/splits.json)')
    parser.add_argument('--epochs', type=int, default=None,
                         help='Override config.epochs, e.g. --epochs 2 for a quick end-to-end smoke run')
    parser.add_argument('--batch_size', type=int, default=None,
                         help='Override config.batch_size (config default is 1, tuned for ~16GB VRAM)')
    parser.add_argument('--no_checkpoint', action='store_true',
                         help='Disable gradient checkpointing (faster, but needs more VRAM -- only if you have >=24GB)')
    parser.add_argument('--num_workers', type=int, default=None,
                         help='Override config.num_workers (default 4) -- parallel CPU data-loading processes')
    parser.add_argument('--max_train_cases', type=int, default=None,
                         help='Cap the number of training cases (val/test stay full-size) -- a speed/compute-budget '
                              'compromise, not from the paper. E.g. --max_train_cases 250 cuts epoch time ~4x '
                              'vs the full ~1000-case train split. State the number actually used in the report.')
    args = parser.parse_args()

    cfg = config
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_checkpoint:
        cfg.model_config['use_checkpoint'] = False
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    if not os.path.isdir(args.data_path):
        raise SystemExit(
            f"data_path '{args.data_path}' not found. Put BraTS cases there as "
            f"<data_path>/<case_id>/<case_id>_flair.nii.gz (+ _t1/_t1ce/_t2/_seg), "
            f"or pass --data_path pointing at your data folder."
        )

    splits_path = args.splits_path or os.path.join(args.checkpoint_dir, 'splits.json')

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: no CUDA GPU detected -- training on CPU will be impractically slow.")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = get_logger('train', os.path.join(args.checkpoint_dir, 'log'))
    log_config_info(cfg, logger)

    splits = make_splits(args.data_path, cfg.train_split, cfg.val_split, cfg.seed, splits_path,
                          max_train_cases=args.max_train_cases)
    split_msg = f"Split sizes -- train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}"
    print(split_msg)
    logger.info(split_msg)

    crop_size = (cfg.input_size_h, cfg.input_size_w, cfg.input_size_c)
    train_dataset = BratsDataset(args.data_path, split='train', case_ids=splits['train'], crop_size=crop_size)
    val_dataset = BratsDataset(args.data_path, split='val', case_ids=splits['val'], crop_size=crop_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = WASMamba(
        input_channels=cfg.model_config['input_channels'],
        num_classes=cfg.model_config['num_classes'],
        depths=cfg.model_config['depths'],
        depths_decoder=cfg.model_config['depths_decoder'],
        drop_path_rate=cfg.model_config['drop_path_rate'],
        use_checkpoint=cfg.model_config.get('use_checkpoint', False),
    ).to(device)

    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    criterion = cfg.criterion

    start_epoch = 1
    best_val_dice = 0.0
    latest_ckpt_path = os.path.join(args.checkpoint_dir, 'latest.pth')
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best.pth')

    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        resume_msg = f"Resumed from {latest_ckpt_path} at epoch {start_epoch} (best_val_dice so far: {best_val_dice:.4f})"
        print(resume_msg)
        logger.info(resume_msg)

    print(f"Class order for all per-class metrics below: {CLASS_NAMES[1:]}")
    logger.info(f"Class order for all per-class metrics below: {CLASS_NAMES[1:]}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss, epoch_ce, epoch_dice_loss = 0.0, 0.0, 0.0
        train_dice_per_class = [[] for _ in range(1, cfg.num_classes)]

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{cfg.epochs}", unit="step")
        for step, batch in pbar:
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Component breakdown for logging only -- criterion's own
            # forward already computed the combined loss used for backward;
            # this re-runs its two sub-losses to report them separately.
            with torch.no_grad():
                ce_val = criterion.celoss(output, label.long()).item()
                dice_val = criterion.diceloss(output, label, softmax=True).item()

            epoch_loss += loss.item()
            epoch_ce += ce_val
            epoch_dice_loss += dice_val

            # Live per-class train Dice on this batch (cheap, same output
            # tensor already computed above -- not a full separate pass).
            with torch.no_grad():
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()
                gt = label.cpu().numpy()
                batch_dice = []
                for c in range(1, cfg.num_classes):
                    d = _dice_score(pred == c, gt == c)
                    train_dice_per_class[c - 1].append(d)
                    batch_dice.append(d)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_val:.4f}',
                'dice_loss': f'{dice_val:.4f}',
                'dice': f'{float(np.mean(batch_dice)):.4f}',  # actual Dice SCORE, not dice_loss (1-Dice)
            })

            if step % cfg.print_interval == 0:
                msg = (f"Epoch {epoch}/{cfg.epochs} Step {step}/{len(train_loader)} "
                       f"loss {loss.item():.4f} (ce {ce_val:.4f} + dice_loss {dice_val:.4f}) "
                       f"| train_dice this batch -- {_format_per_class(batch_dice)}")
                logger.info(msg)  # goes to the log file; tqdm's bar already shows this on screen

        scheduler.step()
        n_steps = max(len(train_loader), 1)
        mean_train_loss = epoch_loss / n_steps
        mean_train_ce = epoch_ce / n_steps
        mean_train_dice_loss = epoch_dice_loss / n_steps
        mean_train_dice_per_class = [float(np.mean(d)) if d else 0.0 for d in train_dice_per_class]

        # Save every epoch, not just periodically -- Colab/Kaggle can
        # disconnect at any time, and re-doing a whole epoch is far more
        # expensive than the few seconds this write costs.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
        }, latest_ckpt_path)

        epoch_time = time.time() - epoch_start
        eta_hours = epoch_time * (cfg.epochs - epoch) / 3600

        summary = (
            f"\n===== Epoch {epoch}/{cfg.epochs} summary "
            f"({epoch_time:.1f}s, ETA {eta_hours:.1f}h for remaining epochs) =====\n"
            f"  train_loss      : {mean_train_loss:.4f}  (ce {mean_train_ce:.4f} + dice_loss {mean_train_dice_loss:.4f})\n"
            f"  train_dice/class: {_format_per_class(mean_train_dice_per_class)}\n"
            f"  train_mean_dice : {float(np.mean(mean_train_dice_per_class)):.4f}\n"
        )

        if epoch % cfg.val_interval == 0:
            val_loss, val_ce, val_dice_loss, dice_per_class, hd95_per_class = validate(
                model, val_loader, criterion, cfg.num_classes, device
            )
            mean_dice = float(np.mean(dice_per_class))
            summary += (
                f"  val_loss        : {val_loss:.4f}  (ce {val_ce:.4f} + dice_loss {val_dice_loss:.4f})\n"
                f"  val_dice/class  : {_format_per_class(dice_per_class)}\n"
                f"  val_mean_dice   : {mean_dice:.4f}\n"
                f"  val_hd95/class  : {_format_per_class(hd95_per_class)} (voxels; n/a = class absent from GT this epoch)\n"
            )

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save(model.state_dict(), best_ckpt_path)
                summary += f"  -> New best mean dice {best_val_dice:.4f} -- saved {best_ckpt_path}\n"

        print(summary)
        logger.info(summary)


if __name__ == '__main__':
    main()
