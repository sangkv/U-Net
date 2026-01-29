import torch
import tifffile as tiff
import numpy as np
import os
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu

# Importing your modules
from unet import UNet
from dataset import EMDataset
from postprocess import apply_threshold, remove_small_regions

def predict(
    volume_path,
    model_path="best_unet.pth",
    save_dir="results",
    threshold_mode="fixed", # "fixed" or "otsu"
    fixed_thresh=0.35,
    min_region_size=100,
    device=None
):
    """
    Comprehensive inference pipeline combining thresholding and post-processing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Starting Inference on: {volume_path} ---")
    print(f"Device: {device} | Mode: {threshold_mode}")

    # 1. Load Model
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare Dataset
    dataset = EMDataset(volume_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    prob_preds = []
    bin_preds = []

    # 3. Inference Loop
    with torch.no_grad():
        for i, img in enumerate(loader):
            img = img.to(device) # (1, 1, H, W)

            # Get probability map
            logits = model(img)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            # 4. Thresholding Strategy
            if threshold_mode == "otsu":
                try:
                    t = threshold_otsu(prob)
                except ValueError: # Handle cases with zero variance
                    t = fixed_thresh
            else:
                t = fixed_thresh

            # Apply threshold
            binary = apply_threshold(prob, t)

            # 5. Post-processing
            if min_region_size > 0:
                binary = remove_small_regions(binary, min_size=min_region_size)

            prob_preds.append(prob.astype(np.float32))
            bin_preds.append((binary * 255).astype(np.uint8))

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} slices...")

    # 6. Save results
    os.makedirs(save_dir, exist_ok=True)
    
    prob_stack = np.stack(prob_preds)
    bin_stack = np.stack(bin_preds)

    prob_save_path = os.path.join(save_dir, "test_probability.tif")
    bin_save_path = os.path.join(save_dir, "test_binary.tif")

    tiff.imwrite(prob_save_path, prob_stack)
    tiff.imwrite(bin_save_path, bin_stack)

    print(f"--- Inference Completed ---")
    print(f"Saved Probability Map: {prob_save_path}")
    print(f"Saved Binary Mask: {bin_save_path}")

if __name__ == "__main__":
    # Example usage:
    # Set threshold_mode="otsu" for automatic thresholding per slice
    # Set threshold_mode="fixed" to use your pre-tuned BEST_T
    predict(
        volume_path="data/test-volume.tif",
        model_path="best_unet.pth",
        threshold_mode="fixed", 
        fixed_thresh=0.35,      # Tuned from tune_threshold.py
        min_region_size=100     # Clean artifacts smaller than this
    )

