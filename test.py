import torch
import tifffile as tiff
import numpy as np
from unet import UNet
from dataset import EMDataset
from skimage.filters import threshold_otsu


@torch.no_grad()
def test(threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load model -----
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    model.load_state_dict(torch.load("best_unet.pth", map_location=device))
    model.eval()

    # ----- dataset -----
    dataset = EMDataset("data/test-volume.tif")

    prob_preds = []
    bin_preds = []

    for img in dataset:
        img = img.unsqueeze(0).to(device)   # (1,1,H,W)

        logits = model(img)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H,W)

        # ----- thresholding -----
        if threshold is None:
            # Otsu per-slice
            t = threshold_otsu(prob)
        else:
            t = threshold

        binary = (prob > t).astype(np.uint8)  # {0,1}

        prob_preds.append(prob)
        bin_preds.append(binary)

    prob_preds = np.stack(prob_preds).astype(np.float32)
    bin_preds = (np.stack(bin_preds) * 255).astype(np.uint8)

    # ----- save -----
    tiff.imwrite("test-probability.tif", prob_preds)
    tiff.imwrite("test-binary.tif", bin_preds)

    print("Saved:")
    print("  - test-probability.tif (float32, [0,1])")
    print("  - test-binary.tif (uint8, 0/255)")


if __name__ == "__main__":
    test()
