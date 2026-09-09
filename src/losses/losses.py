"""
Loss functions used by the WAS-Mamba reference implementation.

Extracted from utils.py, https://github.com/1605066114/WAS-Mamba
(only the loss classes; training/logging helpers moved to src/utils/train_utils.py,
metric/eval helpers moved to src/utils/metrics.py).

NOTE for BraTS: `CeDiceLoss` below is what the repo's own Synapse config
(`config_setting.py`) wires up as `criterion` — and even there its
cross-entropy term is commented out, so effectively it trains with
Dice-only loss. Whether the paper's BraTS experiments used the same
Dice-only setup, or BCE+Dice (`BceDiceLoss` below), or something else,
is NOT stated in the repo — check the paper's text/tables for BraTS
specifically before picking one.
"""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class nDiceLoss(nn.Module):
    """Multi-class (one-hot) Dice loss — this is the one BraTS training will need,
    since BraTS has multiple sub-regions rather than a single foreground mask."""

    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    """This is what the repo's Synapse config actually uses as `criterion`.
    Note the CE term is dead code in the original (commented out) — as
    written, this is really just Dice loss. Left exactly as the authors had
    it; decide deliberately whether to re-enable CE for BraTS."""

    def __init__(self, num_classes, loss_weight=[0, 1]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        # loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        loss = loss_dice
        return loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class PaperDiceCeLoss(nn.Module):
    """The loss the WAS-Mamba PAPER actually states it uses (Eq. 13, Section
    III-E): soft Dice + cross-entropy, summed over classes/voxels.

    This is written from the paper's text, NOT copied from the repo — the
    repo's own `CeDiceLoss` above has its CE term commented out (so it's
    silently Dice-only), which does not match what the paper describes.
    For BraTS specifically, use this class, not `CeDiceLoss`, to match the
    paper's stated setup.
    """

    def __init__(self, num_classes):
        super(PaperDiceCeLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target.long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        return loss_ce + loss_dice


class GT_BceDiceLoss(nn.Module):
    """Deep-supervision variant: applies BceDiceLoss to the final output plus
    4 intermediate decoder outputs, with fixed weights 0.1/0.2/0.3/0.4/0.5.
    Only usable if the model actually returns those 5 outputs — WASMamba's
    forward() in this repo returns a single tensor, so this loss does not
    apply to it as-is."""

    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss
