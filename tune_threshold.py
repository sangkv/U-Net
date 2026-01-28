import torch
import numpy as np
from torch.utils.data import DataLoader

from unet import UNet
from dataset import EMDataset
from metrics import dice_score
from postprocess import apply_threshold


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load model -----
    model = UNet(1, 1, bilinear=True).to(device)
    model.load_state_dict(torch.load("best_unet.pth", map_location=device))
    model.eval()

    # ----- validation dataset -----
    TRAIN_VOLUME = "data/train-volume.tif"
    TRAIN_LABEL = "data/train-labels.tif"
    val_ds = EMDataset(
        TRAIN_VOLUME,
        TRAIN_LABEL
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    thresholds = np.linspace(0.3, 0.7, 17)

    print("Threshold tuning:")
    for t in thresholds:
        dice_sum = 0.0

        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            prob = torch.sigmoid(model(img)).cpu().numpy()[0, 0]
            pred = apply_threshold(prob, t)
            pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)

            dice_sum += dice_score(pred, mask).item()

        print(f"  t={t:.2f} | Dice={dice_sum/len(loader):.4f}")


if __name__ == "__main__":
    main()
