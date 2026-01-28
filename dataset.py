import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np


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
