import torch
import tifffile as tiff
import numpy as np
from unet import UNet
from dataset import EMSegmentationDataset


@torch.no_grad()
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    model.load_state_dict(torch.load("unet_em.pth", map_location=device))
    model.eval()

    dataset = EMSegmentationDataset("data/test-volume.tif")

    preds = []

    for img in dataset:
        img = img.unsqueeze(0).to(device)
        pred = model(img)
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()
        preds.append(pred)

    preds = np.stack(preds)
    tiff.imwrite("test-predictions.tif", preds.astype(np.float32))


if __name__ == "__main__":
    test()
