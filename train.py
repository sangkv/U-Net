import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam

from unet import UNet
from dataset import EMDataset
from config import Config
from utils.augment import EMAugmentor
from utils.losses import UnifiedLoss
from utils.metrics import SegmentationMetrics


def train():
    # ----------------------------------------------------------
    # 1. Deterministic train / validation split
    # ----------------------------------------------------------
    train_idx, val_idx = Config.get_split_indices()
    print(f"Train slices: {train_idx}")
    print(f"Val slices:   {val_idx}")

    # ----------------------------------------------------------
    # 2. Augmentor (EM-safe, config-driven)
    # ----------------------------------------------------------
    augmentor = EMAugmentor(
        p_elastic=Config.AUGMENT_P_ELASTIC,
        alpha_range=Config.AUGMENT_ALPHA_RANGE,
        sigma_range=Config.AUGMENT_SIGMA_RANGE,
    )

    # ----------------------------------------------------------
    # 3. Datasets
    # ----------------------------------------------------------
    train_ds = EMDataset(
        volume_path=Config.VOLUME_PATH,
        label_path=Config.LABEL_PATH,
        indices=train_idx,
        patch_size=Config.PATCH_SIZE,
        patches_per_slice=Config.PATCHES_PER_SLICE,
        augmentor=augmentor,
    )

    val_ds = EMDataset(
        volume_path=Config.VOLUME_PATH,
        label_path=Config.LABEL_PATH,
        indices=val_idx,
    )

    # ----------------------------------------------------------
    # 4. DataLoaders
    # ----------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # ----------------------------------------------------------
    # 5. Model
    # ----------------------------------------------------------
    model = UNet(
        n_channels=1,
        n_classes=1,
        bilinear=True,
    ).to(Config.DEVICE)

    # ----------------------------------------------------------
    # 6. Loss & optimizer (semantic training)
    # ----------------------------------------------------------
    criterion = UnifiedLoss(
        bce_weight=Config.LOSS_BCE_WEIGHT,
        dice_weight=Config.LOSS_DICE_WEIGHT,
        edge_weight=Config.LOSS_EDGE_WEIGHT,
        pos_weight=Config.LOSS_POS_WEIGHT,
    )

    optimizer = Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
    )

    evaluator = SegmentationMetrics(device=Config.DEVICE)

    # ----------------------------------------------------------
    # 7. Training loop
    # ----------------------------------------------------------
    best_dice = 0.0
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_losses = []

        for images, masks in train_loader:
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # ------------------------------------------------------
        # 8. Validation (pixel-level ONLY)
        # ------------------------------------------------------
        model.eval()
        val_dices = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(Config.DEVICE, non_blocking=True)
                masks = masks.to(Config.DEVICE, non_blocking=True)

                logits = model(images)
                metrics = evaluator.get_pixel_metrics(logits, masks)
                val_dices.append(metrics["dice"])

        avg_val_dice = float(np.mean(val_dices))

        print(
            f"Epoch [{epoch:03d}/{Config.EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        # ------------------------------------------------------
        # 9. Checkpoint
        # ------------------------------------------------------
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(">>> Best model saved")

    print(f"\nTraining completed. Best Val Dice = {best_dice:.4f}")


if __name__ == "__main__":
    train()
