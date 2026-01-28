import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects


def apply_threshold(prob, thresh):
    return (prob > thresh).astype(np.uint8)


def remove_small_regions(mask, min_size=100):
    labeled = label(mask)
    cleaned = remove_small_objects(labeled, min_size=min_size)
    return (cleaned > 0).astype(np.uint8)
