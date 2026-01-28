import torch
import tifffile as tiff
import numpy as np
from torch.utils.data import DataLoader

from unet import UNet
from dataset import EMDataset
from postprocess import apply_threshold, remove_small_regions


BEST_T = 0.35        # ← retrieved from tune_threshold
MIN_SIZE = 100


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load model -----
    model = UNet(1, 1, bilinear=True).to(device)
    model.load_state_dict(torch.load("best_unet.pth", map_location=device))
    model.eval()

    # ----- dataset -----
    test_ds = EMDataset("data/test-volume.tif")
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    prob_preds = []
    bin_preds = []

    for img in loader:
        img = img.to(device)

        prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()
        mask = apply_threshold(prob, BEST_T)
        mask = remove_small_regions(mask, min_size=MIN_SIZE)

        prob_preds.append(prob.astype(np.float32))
        bin_preds.append(mask.astype(np.uint8))

    prob_preds = np.stack(prob_preds)
    bin_preds = np.stack(bin_preds)

    # ----- save -----
    tiff.imwrite("test_probability.tif", prob_preds)
    tiff.imwrite("test_binary.tif", bin_preds)

    print("Saved:")
    print(" - test_probability.tif  (float32, [0,1])")
    print(" - test_binary.tif       (uint8, {0,1})")


if __name__ == "__main__":
    main()
