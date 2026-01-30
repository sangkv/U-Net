import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import random


class EMDataset(Dataset):
    """
    Dataset for Electron Microscopy (EM) image stacks.

    Supports:
    - Slice-based sampling
    - Random patch extraction
    - Optional data augmentation
    """

    def __init__(
        self,
        volume_path,
        label_path=None,
        indices=None,
        patch_size=None,
        patches_per_slice=1,
        augmentor=None
    ):
        self.volume = tiff.imread(volume_path).astype(np.float32)

        if self.volume.ndim != 3:
            raise ValueError("Expected volume with shape (D, H, W).")

        v_min = np.min(self.volume)
        v_max = np.max(self.volume)
        self.volume = (self.volume - v_min) / (v_max - v_min + 1e-8)

        self.label = None
        if label_path is not None:
            self.label = tiff.imread(label_path).astype(np.float32)
            if self.label.shape != self.volume.shape:
                raise ValueError("Volume and label must have the same shape.")
            self.label = (self.label > 127).astype(np.float32)

        self.indices = indices if indices is not None else list(range(len(self.volume)))
        self.patch_size = patch_size
        self.patches_per_slice = int(patches_per_slice)
        self.augmentor = augmentor

    def __len__(self):
        return len(self.indices) * self.patches_per_slice

    def _random_crop(self, img, mask):
        h, w = img.shape
        ps = self.patch_size

        if ps > h or ps > w:
            raise ValueError(
                f"patch_size ({ps}) larger than image size ({h}, {w})."
            )

        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)

        img = img[y:y + ps, x:x + ps]
        if mask is not None:
            mask = mask[y:y + ps, x:x + ps]

        return img, mask

    def __getitem__(self, idx):
        slice_idx = self.indices[idx // self.patches_per_slice]

        img = self.volume[slice_idx]
        mask = self.label[slice_idx] if self.label is not None else None

        # 1. Random patch extraction
        if self.patch_size is not None:
            img, mask = self._random_crop(img, mask)

        # 2. Data augmentation
        if self.augmentor is not None and mask is not None:
            img, mask = self.augmentor(img, mask)

        # 3. Convert to tensors
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        if mask is not None:
            mask = (mask > 0.5).astype(np.float32)
            mask = np.ascontiguousarray(mask)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            return img_tensor, mask_tensor

        return img_tensor
