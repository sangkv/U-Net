import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates


class EMAugmentor:
    """
    Advanced augmentation suite for Electron Microscopy (EM) images.

    Augmentations:
    - Random flips
    - Random 90-degree rotations
    - Elastic deformation (U-Net style)

    Designed to preserve topology while introducing realistic variability.
    """

    def __init__(self, p_elastic=0.3, alpha_range=(3, 8), sigma_range=(6, 10)):
        self.p_elastic = float(p_elastic)
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def _elastic_deform(self, img, mask, alpha, sigma):
        """
        Apply elastic deformation to a single-channel image and mask.
        """
        if img.ndim != 2 or mask.ndim != 2:
            raise ValueError("Elastic deformation expects 2D arrays (H, W).")

        h, w = img.shape
        random_state = np.random.RandomState()

        # Scale deformation relative to image size
        alpha = alpha * min(h, w) / 100.0
        sigma = sigma * min(h, w) / 100.0

        dx = gaussian_filter(
            (random_state.rand(h, w) * 2 - 1),
            sigma,
            mode="reflect"
        ) * alpha

        dy = gaussian_filter(
            (random_state.rand(h, w) * 2 - 1),
            sigma,
            mode="reflect"
        ) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (y + dy, x + dx)

        img_deformed = map_coordinates(
            img,
            indices,
            order=1,
            mode="reflect"
        )

        mask_deformed = map_coordinates(
            mask,
            indices,
            order=0,
            mode="reflect"
        )

        return img_deformed, mask_deformed

    def __call__(self, img, mask):
        """
        Apply augmentations to image and mask.

        Args:
            img (ndarray): 2D image array (H, W)
            mask (ndarray): 2D mask array (H, W)

        Returns:
            tuple: Augmented (img, mask), both contiguous arrays
        """
        img = np.asarray(img)
        mask = np.asarray(mask)

        if img.ndim != 2 or mask.ndim != 2:
            raise ValueError("EMAugmentor expects 2D image and mask.")

        # Ensure float image, binary mask
        img = img.astype(np.float32, copy=False)
        mask = (mask > 0).astype(np.float32, copy=False)

        # 1. Random flips
        if random.random() < 0.5:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)

        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)

        # 2. Random 90-degree rotation
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)

        # 3. Elastic deformation
        if random.random() < self.p_elastic:
            alpha = random.uniform(*self.alpha_range)
            sigma = random.uniform(*self.sigma_range)
            img, mask = self._elastic_deform(img, mask, alpha, sigma)

        return np.ascontiguousarray(img), np.ascontiguousarray(mask)
