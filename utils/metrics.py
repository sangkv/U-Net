import torch
import numpy as np
from skimage.measure import label
from skimage.metrics import adapted_rand_error, contingency_table
from skimage.segmentation import relabel_sequential


class SegmentationMetrics:
    """
    Collection of pixel-level and structural metrics for segmentation tasks.

    Supports:
    - Binary semantic segmentation
    - Instance-aware evaluation (via connected components)

    All metrics are computed per-image and averaged over the batch.
    """

    def __init__(self, device='cpu'):
        self.device = device

    @torch.no_grad()
    def get_pixel_metrics(self, pred_logits, target, threshold=0.5):
        """
        Compute pixel-level Dice and IoU metrics.

        Args:
            pred_logits (Tensor): Model output logits of shape (B, 1, H, W)
            target (Tensor): Ground truth mask of shape (B, 1, H, W)
            threshold (float): Threshold applied after sigmoid

        Returns:
            dict: Mean Dice and IoU over the batch
        """
        prob = torch.sigmoid(pred_logits)
        pred = (prob > threshold).float()

        B = pred.shape[0]
        pred_f = pred.view(B, -1)
        tgt_f = target.view(B, -1)

        intersection = (pred_f * tgt_f).sum(dim=1)
        union = pred_f.sum(dim=1) + tgt_f.sum(dim=1)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)

        return {
            "dice": dice.mean().item(),
            "iou": iou.mean().item()
        }

    def get_structural_metrics(self, pred_mask, target_mask):
        """
        Compute structural metrics (Adapted Rand Error and Variation of Information).

        Args:
            pred_mask (ndarray): Binary prediction mask, shape (H, W) or (B, H, W)
            target_mask (ndarray): Binary target mask, shape (H, W) or (B, H, W)

        Returns:
            dict: Mean Rand Error, VI total, VI split, VI merge
        """

        def to_instances(mask):
            inst = label(mask > 0.5)
            inst, _, _ = relabel_sequential(inst)
            return inst.astype(np.int32)

        pred_mask = np.asarray(pred_mask)
        target_mask = np.asarray(target_mask)

        if pred_mask.ndim == 2:
            pred_mask = pred_mask[None, ...]
            target_mask = target_mask[None, ...]

        results = []

        for b in range(pred_mask.shape[0]):
            p_inst = to_instances(pred_mask[b])
            t_inst = to_instances(target_mask[b])

            if p_inst.max() == 0 and t_inst.max() == 0:
                results.append({
                    "rand_error": 0.0,
                    "vi_total": 0.0,
                    "vi_split": 0.0,
                    "vi_merge": 0.0
                })
                continue

            rand_err, _, _ = adapted_rand_error(
                t_inst, p_inst, ignore_labels=(0,)
            )

            table = contingency_table(t_inst, p_inst)
            table = table.toarray().astype(np.float64)

            p_ij = table / (table.sum() + 1e-8)
            p_i = p_ij.sum(axis=1)
            p_j = p_ij.sum(axis=0)

            def entropy(p):
                p = p[p > 0]
                return -np.sum(p * np.log2(p))

            h_joint = entropy(p_ij.ravel())
            h_t = entropy(p_i)
            h_p = entropy(p_j)

            vi_split = h_joint - h_p      # H(T | P)
            vi_merge = h_joint - h_t      # H(P | T)

            results.append({
                "rand_error": float(rand_err),
                "vi_total": float(vi_split + vi_merge),
                "vi_split": float(vi_split),
                "vi_merge": float(vi_merge)
            })

        keys = results[0].keys()
        return {
            k: float(np.mean([r[k] for r in results]))
            for k in keys
        }

    def get_all_metrics(self, pred_logits, target_mask):
        """
        Compute both pixel-level and structural metrics.

        Args:
            pred_logits (Tensor): Model output logits of shape (B, 1, H, W)
            target_mask (Tensor): Ground truth mask of shape (B, 1, H, W)

        Returns:
            dict: Combined metrics dictionary
        """
        pixel_res = self.get_pixel_metrics(pred_logits, target_mask)

        pred_np = (torch.sigmoid(pred_logits) > 0.5).detach().cpu().numpy()
        tgt_np = target_mask.detach().cpu().numpy()

        if pred_np.ndim == 4:
            pred_np = pred_np[:, 0]
        if tgt_np.ndim == 4:
            tgt_np = tgt_np[:, 0]

        struct_res = self.get_structural_metrics(pred_np, tgt_np)
        return {**pixel_res, **struct_res}
