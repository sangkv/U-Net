import numpy as np
from skimage.measure import label
from skimage.metrics import adapted_rand_error
from skimage.segmentation import relabel_sequential
from scipy.stats import entropy


def membrane_to_neuron_instances(mem_mask):
    """
    mem_mask: binary membrane mask {0,1}
    return: labeled neuron instances
    """
    neuron = 1 - mem_mask
    labeled = label(neuron)
    labeled, _, _ = relabel_sequential(labeled)
    return labeled


def rand_error(pred_inst, gt_inst):
    """
    Foreground Rand Error
    """
    rand_err, _, _ = adapted_rand_error(
        gt_inst, pred_inst, ignore_labels=(0,)
    )
    return rand_err


def variation_of_information(pred_inst, gt_inst):
    """
    VI = VI_split + VI_merge
    """
    vi_split, vi_merge = _vi_components(pred_inst, gt_inst)
    return vi_split + vi_merge, vi_split, vi_merge


def _vi_components(seg, gt):
    seg = seg.flatten()
    gt = gt.flatten()

    # contingency table
    contingency = {}
    for s, g in zip(seg, gt):
        if g == 0:
            continue
        contingency.setdefault((s, g), 0)
        contingency[(s, g)] += 1

    seg_sizes = {}
    gt_sizes = {}

    for (s, g), v in contingency.items():
        seg_sizes[s] = seg_sizes.get(s, 0) + v
        gt_sizes[g] = gt_sizes.get(g, 0) + v

    total = sum(contingency.values())

    vi_split = 0.0
    vi_merge = 0.0

    for (s, g), v in contingency.items():
        p = v / total
        vi_split += p * np.log(gt_sizes[g] / v)
        vi_merge += p * np.log(seg_sizes[s] / v)

    return vi_split, vi_merge
