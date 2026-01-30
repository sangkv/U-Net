import numpy as np
import torch
import random
from pathlib import Path


class Config:
    """
    Central configuration for EM semantic / instance segmentation pipeline.

    This config is designed for:
    - Slice-based EM volumes (.tif stacks)
    - Semantic training (U-Net, BCE + Dice)
    - Instance-level evaluation via post-processing
    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    SEED = 42

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    DATA_DIR = Path("data")

    VOLUME_PATH = DATA_DIR / "train-volume.tif"
    LABEL_PATH = DATA_DIR / "train-labels.tif"
    TEST_VOLUME_PATH = DATA_DIR / "test-volume.tif"

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_PATH = CHECKPOINT_DIR / "best_unet.pth"

    EVAL_DIR = Path("eval")

    # ------------------------------------------------------------------
    # Dataset / Split configuration
    # ------------------------------------------------------------------
    # IMPORTANT:
    # This must match the number of slices in the TIFF volume.
    # It is kept explicit for reproducibility and experiment tracking.
    TOTAL_SLICES = 30

    VAL_RATIO = 0.2

    @classmethod
    def get_split_indices(cls):
        """
        Deterministic train / validation split.

        This method MUST be used everywhere (train / eval)
        to guarantee consistent comparison across experiments.
        """
        indices = np.arange(cls.TOTAL_SLICES)
        rng = np.random.RandomState(cls.SEED)
        rng.shuffle(indices)

        split_point = int(cls.TOTAL_SLICES * (1.0 - cls.VAL_RATIO))
        train_idx = indices[:split_point].tolist()
        val_idx = indices[split_point:].tolist()

        return train_idx, val_idx

    # ------------------------------------------------------------------
    # Training hyperparameters (semantic segmentation)
    # ------------------------------------------------------------------
    BATCH_SIZE = 4

    PATCH_SIZE = 256
    PATCHES_PER_SLICE = 10

    LEARNING_RATE = 1e-4
    EPOCHS = 200

    # ------------------------------------------------------------------
    # Augmentation (EM-specific)
    # ------------------------------------------------------------------
    # These values are chosen to preserve topology
    # while increasing local deformation variability.
    AUGMENT_P_ELASTIC = 0.5
    AUGMENT_ALPHA_RANGE = (3.0, 8.0)
    AUGMENT_SIGMA_RANGE = (6.0, 10.0)

    # ------------------------------------------------------------------
    # Loss configuration
    # ------------------------------------------------------------------
    # Edge weighting emphasizes thin membrane structures.
    LOSS_BCE_WEIGHT = 1.0
    LOSS_DICE_WEIGHT = 3.0
    LOSS_EDGE_WEIGHT = 5.0
    LOSS_POS_WEIGHT = 1.0

    # ------------------------------------------------------------------
    # Evaluation / Post-processing (instance-level)
    # ------------------------------------------------------------------
    # These parameters are NOT model parameters.
    # They belong to the post-processing pipeline.
    EVAL_THRESHOLDS = np.linspace(0.3, 0.8, 11).tolist()
    EVAL_MIN_SIZES = [0, 50, 100, 200, 500]

    # Post-processing (EM-safe defaults)
    POSTPROCESS_MIN_SIZE = 100
    POSTPROCESS_OPENING_RADIUS = 0  # Disabled by default (EM-safe)

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Global seeding (must be executed exactly once)
# ----------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(Config.SEED)
