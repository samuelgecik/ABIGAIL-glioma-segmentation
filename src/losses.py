"""Loss functions for binary segmentation, with a small factory.

Centralizes loss construction so training can switch between BCE (with an
optional positive-class weight), soft Dice, or a BCE+Dice combo from the CLI.

Why this matters here: BCE with a large ``pos_weight`` (auto-computed as the
negative/positive pixel ratio, ~96 for this dataset) rescues the rare tumor
class from being ignored, but it inflates predicted probabilities and pushes
the Dice-optimal threshold up toward ~1.0 (poor calibration). A milder
``pos_weight`` or a Dice/BCE+Dice loss keeps the optimum near 0.5 and yields
more usable probability maps.
"""
import torch
import torch.nn as nn

VALID_LOSSES = ("bce", "dice", "bce_dice")


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (operates on raw logits).

    Dice is computed per-sample over the flattened spatial dims, then averaged
    over the batch. The ``smooth`` term avoids division by zero on empty masks
    and provides a gradient when prediction and target are both near-empty.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).to(probs.dtype)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice.

    Combines BCE's pixel-wise gradient (good early-training signal) with Dice's
    direct overlap optimization (handles imbalance, better-calibrated optimum).
    """

    def __init__(self, pos_weight=None, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(name: str, pos_weight=None) -> nn.Module:
    """Construct a loss by name: 'bce', 'dice', or 'bce_dice'.

    ``pos_weight`` (a tensor) is used by the BCE component; pure Dice ignores it.
    """
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss '{name}'. Choose from {VALID_LOSSES}.")
