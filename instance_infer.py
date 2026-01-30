# instance_infer.py
import os
import json
import torch
import numpy as np
import tifffile as tiff

from unet import UNet
from dataset import EMDataset
from config import Config
from utils.postprocess import EMPostProcessor, apply_threshold


def instance_infer():
    print("=== Instance Inference (pipeline-centric) ===")

    # --------------------------------------------------
    # 1. Load best pipeline from evaluation
    # --------------------------------------------------
    report_path = Config.EVAL_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "evaluation_report.json not found. "
            "Run evaluate.py before instance inference."
        )

    with open(report_path, "r") as f:
        report = json.load(f)

    pp_cfg = report["best_instance_pipeline"]["postprocess"]

    threshold = pp_cfg["threshold"]
    min_size = pp_cfg["min_size"]
    opening_radius = pp_cfg["opening_radius"]

    print("Using best instance pipeline:")
    print(f"  threshold       = {threshold}")
    print(f"  min_size        = {min_size}")
    print(f"  opening_radius  = {opening_radius}")

    # --------------------------------------------------
    # 2. Load model
    # --------------------------------------------------
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(
        torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # --------------------------------------------------
    # 3. Dataset
    # --------------------------------------------------
    test_ds = EMDataset(
        Config.TEST_VOLUME_PATH,
        label_path=None
    )

    processor = EMPostProcessor(
        min_size=min_size,
        opening_radius=opening_radius
    )

    instance_maps = []

    # --------------------------------------------------
    # 4. Inference + post-processing
    # --------------------------------------------------
    with torch.no_grad():
        for i in range(len(test_ds)):
            img = test_ds[i].unsqueeze(0).to(Config.DEVICE)
            prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()

            # Semantic → binary
            binary = apply_threshold(prob, threshold)

            # Clean semantic mask
            clean = processor.clean_mask(binary)

            # Instance separation
            instances = processor.get_instances(clean)

            instance_maps.append(instances.astype(np.int32))

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_ds)} slices")

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------
    os.makedirs("results", exist_ok=True)

    tiff.imwrite(
        "results/test_instances.tif",
        np.stack(instance_maps)
    )

    print("=== Done ===")
    print("Saved:")
    print(" - results/test_instances.tif  (instance labels)")


if __name__ == "__main__":
    instance_infer()
