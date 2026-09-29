import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Soft Dice + BCE on sigmoid outputs, averaged over the WT/TC/ET channels."""

    def __init__(self, smooth=1e-5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3, 4)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return (1 - dice.mean()) + self.bce_weight * self.bce(logits, target)


@torch.no_grad()
def dice_per_region(logits, target, threshold=0.5, smooth=1e-5):
    """Returns Dice for each output channel (WT, TC, ET)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (0, 2, 3, 4)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return ((2 * inter + smooth) / (denom + smooth)).tolist()


@torch.no_grad()
def sensitivity_per_region(logits, target, threshold=0.5, smooth=1e-5):
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (0, 2, 3, 4)
    tp = (pred * target).sum(dims)
    return ((tp + smooth) / (target.sum(dims) + smooth)).tolist()
