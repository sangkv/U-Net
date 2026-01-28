import torch


@torch.no_grad()
def dice_score(pred, target, threshold=0.5, eps=1e-6):
    """
    pred: logits, shape (B, 1, H, W)
    target: binary mask {0,1}, same shape
    """
    # logits -> probability
    pred = torch.sigmoid(pred)

    # probability -> binary mask
    pred = (pred > threshold).float()

    # per-image Dice
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2 * inter + eps) / (union + eps)

    return dice.mean()
