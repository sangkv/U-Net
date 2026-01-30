import os
import torch
import numpy as np
import tifffile as tiff

from unet import UNet
from dataset import EMDataset
from config import Config


def predict():
    print("=== Semantic Prediction (semantic-only) ===")

    # --------------------------------------------------
    # 1. Load model
    # --------------------------------------------------
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(
        torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # --------------------------------------------------
    # 2. Load test dataset (NO labels)
    # --------------------------------------------------
    test_ds = EMDataset(
        Config.TEST_VOLUME_PATH,
        label_path=None
    )

    probs = []
    binaries = []

    # --------------------------------------------------
    # 3. Inference
    # --------------------------------------------------
    with torch.no_grad():
        for i in range(len(test_ds)):
            img = test_ds[i].unsqueeze(0).to(Config.DEVICE)  # (1,1,H,W)
            prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()

            probs.append(prob.astype(np.float32))

            # Visualization-only binary (NOT optimal, NOT instance)
            bin_vis = (prob > 0.5).astype(np.uint8) * 255
            binaries.append(bin_vis)

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_ds)} slices")

    # --------------------------------------------------
    # 4. Save results
    # --------------------------------------------------
    os.makedirs("results", exist_ok=True)

    tiff.imwrite("results/test_probability.tif", np.stack(probs))
    tiff.imwrite("results/test_binary_t05.tif", np.stack(binaries))

    print("=== Done ===")
    print("Saved:")
    print(" - results/test_probability.tif  (semantic output)")
    print(" - results/test_binary_t05.tif   (visualization only)")


if __name__ == "__main__":
    predict()
