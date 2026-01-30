import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedLoss(nn.Module):
    """
    Hybrid Loss: BCE + Dice + Boundary Weighting.
    Optimized for thin structures and general semantic segmentation.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, edge_weight=2.0, pos_weight=1.0):
        super(UnifiedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        # Use register_buffer for automatic device management (CPU/GPU)
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def _get_edges(self, target):
        """
        Morphological Gradient (Dilation - Erosion) to find boundaries.
        """
        # Kernel 3x3 is safe for both thin membranes and large object edges
        dilation = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        erosion = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        return dilation - erosion

    def dice_loss(self, pred_logits, target, smooth=1e-6):
        """
        Per-image Dice Loss to ensure balanced contribution from each sample.
        """
        probs = torch.sigmoid(pred_logits)
        
        # Calculate over spatial dimensions (H, W) per image in batch
        # Safer than global flatten when batch has empty/small masks
        dims = (2, 3) if probs.dim() == 4 else (1, 2)
        intersection = (probs * target).sum(dim=dims)
        cardinality = (probs + target).sum(dim=dims)
        
        dice_score = (2. * intersection + smooth) / (cardinality + smooth)
        return (1. - dice_score).mean()

    def forward(self, pred_logits, target):
        # 1. Weighted BCE with pre-registered pos_weight
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, 
            pos_weight=self.pos_weight, 
            reduction='none'
        )
        
        # 2. Boundary Weighting (Active if edge_weight > 1.0)
        if self.edge_weight > 1.0:
            with torch.no_grad():
                edge_mask = self._get_edges(target)
            # Create importance map: 1.0 everywhere, higher on edges
            weight_map = 1.0 + (edge_mask * (self.edge_weight - 1.0))
            weighted_bce = (bce * weight_map).mean()
        else:
            weighted_bce = bce.mean()
        
        # 3. Dice Loss for regional overlap
        dice = self.dice_loss(pred_logits, target)
        
        # Combined Loss
        return (self.bce_weight * weighted_bce) + (self.dice_weight * dice)

