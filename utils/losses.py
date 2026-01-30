import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedLoss(nn.Module):
    """
    Hybrid loss for binary segmentation combining:
    - Weighted BCE (with optional boundary emphasis)
    - Dice loss (per-image)

    Designed for UNet-style architectures and thin-structure segmentation.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, edge_weight=2.0, pos_weight=1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.edge_weight = float(edge_weight)

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

    def _get_edges(self, target):
        """
        Compute a morphological gradient (dilation - erosion)
        to approximate object boundaries.
        """
        dilation = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        erosion = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        edge = dilation - erosion
        return edge.clamp(0.0, 1.0)

    def dice_loss(self, pred_logits, target, smooth=1e-6):
        """
        Per-image Dice loss.
        Images with empty target masks are ignored.
        """
        probs = torch.sigmoid(pred_logits)

        dims = (2, 3)
        intersection = (probs * target).sum(dim=dims)
        cardinality = (probs + target).sum(dim=dims)

        dice = (2.0 * intersection + smooth) / (cardinality + smooth)

        valid = target.sum(dim=dims) > 0
        if valid.any():
            return (1.0 - dice[valid]).mean()
        else:
            return torch.zeros((), device=pred_logits.device)

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits (Tensor): shape (B, 1, H, W)
            target (Tensor): shape (B, 1, H, W)
        """
        pos_weight = self.pos_weight.to(dtype=pred_logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            pred_logits,
            target,
            pos_weight=pos_weight,
            reduction="none"
        )

        if self.edge_weight > 1.0:
            with torch.no_grad():
                edge_mask = self._get_edges(target)
            weight_map = 1.0 + edge_mask * (self.edge_weight - 1.0)
            bce = (bce * weight_map).mean()
        else:
            bce = bce.mean()

        dice = self.dice_loss(pred_logits, target)

        return self.bce_weight * bce + self.dice_weight * dice
