import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import random
from augmentations import em_augmentation


class EMDataset(Dataset):
    def __init__(self, volume_path, label_path=None, indices=None):
        self.volume = tiff.imread(volume_path)  # (Z, H, W)
        self.label = None
        if label_path is not None:
            self.label = tiff.imread(label_path)

        if indices is None:
            self.indices = list(range(self.volume.shape[0]))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        z = self.indices[idx]

        img = self.volume[z].astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        if self.label is None:
            return img

        mask = (self.label[z] > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


class EMPatchDataset(Dataset):
    def __init__(
        self,
        volume_path,
        label_path,
        indices,
        patch_size=256,
        patches_per_slice=10,
        augment=True
    ):
        """
        volume: (Z, H, W)
        label:  (Z, H, W)
        """
        self.volume = tiff.imread(volume_path)
        self.label = tiff.imread(label_path)

        self.indices = indices
        self.patch_size = patch_size
        self.patches_per_slice = patches_per_slice
        self.augment = augment

        self.H = self.volume.shape[1]
        self.W = self.volume.shape[2]

    def __len__(self):
        # Each slice generates multiple patches.
        return len(self.indices) * self.patches_per_slice

    def __getitem__(self, idx):
        # map idx → slice
        slice_idx = self.indices[idx // self.patches_per_slice]

        img = self.volume[slice_idx]
        mask = self.label[slice_idx]

        ps = self.patch_size

        # random crop
        y = random.randint(0, self.H - ps)
        x = random.randint(0, self.W - ps)

        img_patch = img[y:y+ps, x:x+ps]
        mask_patch = mask[y:y+ps, x:x+ps]

        if self.augment:
            img_patch, mask_patch = em_augmentation(img_patch, mask_patch)

        # normalize
        img_patch = img_patch.astype(np.float32)
        img_patch = (img_patch - img_patch.min()) / (
            img_patch.max() - img_patch.min() + 1e-8
        )

        img_patch = torch.from_numpy(img_patch).unsqueeze(0)

        mask_patch = (mask_patch > 0).astype(np.float32)
        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0)

        return img_patch, mask_patch
