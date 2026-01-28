import torch
import torch.nn.functional as F


def dice_loss(pred, target, eps=1.0):
    """
    Dice loss computed only on patches that contain positive pixels.
    This avoids rewarding the model on empty (all-background) patches,
    which are common in EM segmentation.
    
    pred: logits, shape (B, 1, H, W)
    target: binary mask {0,1}, same shape
    """
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)

    # Select only patches that contain boundary pixels
    valid = target.sum(dim=(2, 3)) > 0

    if valid.any():
        return 1.0 - dice[valid].mean()
    else:
        # All patches are background; Dice does not contribute
        return torch.tensor(0.0, device=pred.device)


def bce_dice_loss(pred, target, pos_weight=5.0):
    """
    Combined BCE + Dice loss for EM segmentation.
    
    - BCE (with pos_weight) handles extreme class imbalance.
    - Dice focuses learning on boundary regions.
    """
    bce = F.binary_cross_entropy_with_logits(
        pred,
        target,
        pos_weight=torch.tensor([pos_weight], device=pred.device)
    )

    dice = dice_loss(pred, target)

    return bce + dice
