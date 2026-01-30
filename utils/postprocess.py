import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


class EMPostProcessor:
    """
    Post-processing utilities for EM segmentation.

    Design principles:
    - Preserve topology (avoid merge errors).
    - Do NOT fill gaps aggressively (no default closing).
    - Instance separation is performed explicitly via watershed.
    """

    def __init__(self, min_size=0, opening_radius=0):
        """
        Args:
            min_size (int):
                Minimum object size to keep (in pixels).
                Used to remove small isolated noise.
                Set to 0 to disable.

            opening_radius (int):
                Radius for binary opening.
                Used ONLY to remove tiny spurs / noise.
                Safer than closing for EM.
                Set to 0 to disable.
        """
        self.min_size = int(min_size)
        self.opening_radius = int(opening_radius)

    # ------------------------------------------------------------------
    # Semantic mask cleaning
    # ------------------------------------------------------------------
    def clean_mask(self, mask):
        """
        Clean a binary semantic mask.

        Args:
            mask (ndarray): binary mask {0,1}

        Returns:
            ndarray: cleaned binary mask {0,1}

        Notes:
        - No hole filling by default (to avoid merge).
        - Only removes noise, never connects objects.
        """
        mask = (mask > 0).astype(np.uint8)

        # 1. Optional binary opening (safer than closing)
        if self.opening_radius > 0:
            mask = binary_opening(mask, disk(self.opening_radius))

        # 2. Remove small isolated components
        if self.min_size > 0:
            labeled = label(mask)
            labeled = remove_small_objects(labeled, min_size=self.min_size)
            mask = (labeled > 0).astype(np.uint8)

        return mask

    # ------------------------------------------------------------------
    # Instance separation
    # ------------------------------------------------------------------
    def get_instances(self, mask, min_distance=20):
        """
        Convert a binary mask into instance labels using watershed.

        Args:
            mask (ndarray): binary mask {0,1}
            min_distance (int): minimum distance between instance seeds

        Returns:
            ndarray: instance label map (0 = background)
        """
        mask = (mask > 0).astype(np.uint8)

        if mask.sum() == 0:
            return np.zeros(mask.shape, dtype=np.int32)

        # Distance transform inside objects
        distance = ndi.distance_transform_edt(mask)

        # Guard against degenerate cases
        if np.max(distance) == 0:
            return np.zeros(mask.shape, dtype=np.int32)

        # Local maxima as instance seeds
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=mask.astype(np.int32)
        )

        if coords.size == 0:
            # Fallback: treat entire object as one instance
            return label(mask).astype(np.int32)

        seed_mask = np.zeros(mask.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers = label(seed_mask)

        # Watershed to split instances
        instances = watershed(
            -distance,
            markers,
            mask=mask
        )

        return instances.astype(np.int32)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def apply_threshold(prob, threshold=0.5):
    """
    Convert probability map to binary mask.

    Args:
        prob (ndarray): probability map [0,1]
        threshold (float): threshold

    Returns:
        ndarray: binary mask {0,1}
    """
    return (prob > threshold).astype(np.uint8)
