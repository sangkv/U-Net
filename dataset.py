import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff


class EMSegmentationDataset(Dataset):
    def __init__(self, volume_path, label_path=None):
        self.volume = tiff.imread(volume_path).astype(np.float32)
        self.label = None

        if label_path is not None:
            self.label = tiff.imread(label_path).astype(np.float32)

        # normalize to [0, 1]
        self.volume = (self.volume - self.volume.min()) / (
            self.volume.max() - self.volume.min() + 1e-8
        )

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, idx):
        img = self.volume[idx][None, :, :]  # (1, H, W)
        img = torch.from_numpy(img)

        if self.label is not None:
            mask = self.label[idx]
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask)[None, :, :]
            return img, mask

        return img
