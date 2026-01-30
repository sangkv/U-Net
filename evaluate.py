import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from unet import UNet
from dataset import EMDataset
from config import Config
from utils.metrics import SegmentationMetrics
from utils.postprocess import EMPostProcessor, apply_threshold


def evaluate():
    # --------------------------------------------------------------
    # 1. Consistent validation split
    # --------------------------------------------------------------
    _, val_idx = Config.get_split_indices()
    print(f"Evaluating on validation slices: {val_idx}")

    # --------------------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------------------
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(
        torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # --------------------------------------------------------------
    # 3. Dataset & loader (NO augmentation)
    # --------------------------------------------------------------
    val_ds = EMDataset(
        Config.VOLUME_PATH,
        Config.LABEL_PATH,
        indices=val_idx
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    evaluator = SegmentationMetrics(device=Config.DEVICE)

    # --------------------------------------------------------------
    # 4. Precompute probabilities & GT (semantic)
    # --------------------------------------------------------------
    probs = []
    gts = []

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(Config.DEVICE)
            prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()
            probs.append(prob)
            gts.append(mask[0, 0].numpy())

    # --------------------------------------------------------------
    # 5. Semantic evaluation (model-centric)
    # --------------------------------------------------------------
    semantic_scores = []

    for p, g in zip(probs, gts):
        p_bin = apply_threshold(p, threshold=0.5)
        g_bin = (g > 0.5).astype(np.uint8)

        m = evaluator.get_pixel_metrics(
            torch.from_numpy(p_bin).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(g_bin).unsqueeze(0).unsqueeze(0)
        )
        semantic_scores.append(m)

    avg_semantic = {
        k: float(np.mean([s[k] for s in semantic_scores]))
        for k in semantic_scores[0]
    }

    print("Semantic evaluation (threshold = 0.5):")
    for k, v in avg_semantic.items():
        print(f"  {k}: {v:.4f}")

    # --------------------------------------------------------------
    # 6. Instance evaluation (pipeline-centric)
    # --------------------------------------------------------------
    thresholds = Config.EVAL_THRESHOLDS
    min_sizes = Config.EVAL_MIN_SIZES

    instance_results = []

    for t in thresholds:
        for ms in min_sizes:
            processor = EMPostProcessor(
                min_size=ms,
                opening_radius=Config.POSTPROCESS_OPENING_RADIUS
            )

            rand_errors = []
            vi_splits = []
            vi_merges = []

            for p, g in zip(probs, gts):
                # Semantic → binary
                p_bin = apply_threshold(p, t)
                g_bin = (g > 0.5).astype(np.uint8)

                # Clean (semantic-safe)
                p_clean = processor.clean_mask(p_bin)
                g_clean = processor.clean_mask(g_bin)

                # Instance separation
                p_inst = processor.get_instances(p_clean)
                g_inst = processor.get_instances(g_clean)

                # Structural metrics (TRUE instance maps)
                m = evaluator.get_structural_metrics(p_inst, g_inst)

                rand_errors.append(m["rand_error"])
                vi_splits.append(m["vi_split"])
                vi_merges.append(m["vi_merge"])

            result = {
                "postprocess": {
                    "threshold": float(t),
                    "min_size": int(ms),
                    "opening_radius": Config.POSTPROCESS_OPENING_RADIUS
                },
                "metrics": {
                    "rand_error": float(np.mean(rand_errors)),
                    "vi_split": float(np.mean(vi_splits)),
                    "vi_merge": float(np.mean(vi_merges)),
                    "vi_total": float(np.mean(vi_splits) + np.mean(vi_merges))
                }
            }

            print(
                f"T={t:.2f} | min_size={ms:4d} | "
                f"RandErr={result['metrics']['rand_error']:.4f}"
            )

            instance_results.append(result)

    # --------------------------------------------------------------
    # 7. Select best pipeline config (by Rand Error)
    # --------------------------------------------------------------
    best_result = min(
        instance_results,
        key=lambda x: x["metrics"]["rand_error"]
    )

    # --------------------------------------------------------------
    # 8. Save evaluation report
    # --------------------------------------------------------------
    os.makedirs(Config.EVAL_DIR, exist_ok=True)

    report = {
        "semantic_evaluation": avg_semantic,
        "best_instance_pipeline": best_result,
        "all_instance_results": instance_results
    }

    report_path = os.path.join(Config.EVAL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\nBest instance-level configuration:")
    print(json.dumps(best_result, indent=4))
    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    evaluate()
