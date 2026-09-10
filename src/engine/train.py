"""
Training loop for WAS-Mamba base model on BraTS.

Written from scratch this session -- the original repo's own training
script (`test.py` in their GitHub release) imports `datasets.dataset`,
`engine_synapse`, and `configs.config_setting_synapse`, none of which exist
in the repo, so it never actually ran even for the authors' own Synapse
setup. There was nothing usable to port for BraTS specifically.

Usage (run from the repo root, e.g. inside a Colab cell after `%cd`):
    python -m src.engine.train \\
        --data_path /content/drive/MyDrive/BraTS2021/train \\
        --checkpoint_dir /content/drive/MyDrive/wasmamba_checkpoints

IMPORTANT: put --checkpoint_dir on Google Drive (a mounted path under
/content/drive/...), not under /content/ directly -- Colab's local disk is
wiped when the session ends or disconnects, so a checkpoint saved only
there is lost. This script resumes automatically from `latest.pth` in
--checkpoint_dir if it exists, which is what makes training survive
Colab's ~12h session limit across multiple runs.

--epochs lets you override config.epochs for a quick short run (e.g.
`--epochs 2`) to confirm the whole loop works before committing to the
paper's real 1000 epochs.
"""

import os
import json
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.wasmamba import WASMamba
from src.data.brats_dataset import BratsDataset
from src.utils.train_utils import set_seed, get_optimizer, get_scheduler, get_logger, log_config_info
from src.configs.wasmamba_config import setting_config as config


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


def make_splits(data_path, train_split, val_split, seed, splits_path):
    """
    Reproducible 80:5:15 (paper's ratios) train/val/test case-ID split.

    Written once to `splits_path` and reused on every later call/resume, so
    which cases land in which split is fixed and inspectable -- not
    re-randomized on every run (which would silently leak "test" cases into
    training across restarts).
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

    splits = {
        'train': case_ids[:n_train],
        'val': case_ids[n_train:n_train + n_val],
        'test': case_ids[n_train + n_val:],
    }
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    return splits


def validate(model, val_loader, criterion, num_classes, device):
    """
    Per-class Dice on the val split. Unlike src/utils/metrics.py's
    `test_single_volume` (which is Synapse's 2D-slice-based evaluation),
    this operates directly on the 3D volumes BratsDataset returns --
    written fresh for BraTS rather than reusing that function.
    """
    model.eval()
    total_loss = 0.0
    dice_per_class = [[] for _ in range(1, num_classes)]
    with torch.no_grad():
        for batch in val_loader:
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            output = model(image)
            loss = criterion(output, label)
            total_loss += loss.item()

            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()
            gt = label.cpu().numpy()
            for c in range(1, num_classes):
                dice = _dice_score(pred == c, gt == c)
                dice_per_class[c - 1].append(dice)

    mean_loss = total_loss / max(len(val_loader), 1)
    mean_dice_per_class = [float(np.mean(d)) if d else 0.0 for d in dice_per_class]
    return mean_loss, mean_dice_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                         help='Folder containing one subfolder per BraTS case')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                         help='Where to save/resume checkpoints -- put this on Drive, not /content')
    parser.add_argument('--splits_path', type=str, default=None,
                         help='Where to save/load the train/val/test case-ID split (default: <checkpoint_dir>/splits.json)')
    parser.add_argument('--epochs', type=int, default=None,
                         help='Override config.epochs, e.g. --epochs 2 for a quick end-to-end smoke run')
    args = parser.parse_args()

    cfg = config
    if args.epochs is not None:
        cfg.epochs = args.epochs

    splits_path = args.splits_path or os.path.join(args.checkpoint_dir, 'splits.json')

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = get_logger('train', os.path.join(args.checkpoint_dir, 'log'))
    log_config_info(cfg, logger)

    splits = make_splits(args.data_path, cfg.train_split, cfg.val_split, cfg.seed, splits_path)
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

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % cfg.print_interval == 0:
                msg = f"Epoch {epoch}/{cfg.epochs} Step {step}/{len(train_loader)} Loss {loss.item():.4f}"
                print(msg)
                logger.info(msg)

        scheduler.step()
        mean_train_loss = epoch_loss / max(len(train_loader), 1)

        # Save every epoch, not just periodically -- Colab can disconnect at
        # any time, and re-doing a whole epoch on a T4 is far more expensive
        # than the few seconds this write costs.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
        }, latest_ckpt_path)

        if epoch % cfg.val_interval == 0:
            val_loss, dice_per_class = validate(model, val_loader, criterion, cfg.num_classes, device)
            mean_dice = float(np.mean(dice_per_class))
            msg = (f"Epoch {epoch} -- train_loss {mean_train_loss:.4f} val_loss {val_loss:.4f} "
                   f"val_dice_per_class {dice_per_class} mean_dice {mean_dice:.4f}")
            print(msg)
            logger.info(msg)

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"New best mean dice {best_val_dice:.4f} -- saved {best_ckpt_path}")


if __name__ == '__main__':
    main()
