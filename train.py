import torch
from torch.utils.data import DataLoader
from unet import UNet
from dataset import EMDataset
from losses import bce_dice_loss
from metrics import dice_score
import numpy as np


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- split train / val -----
    TRAIN_VOLUME = "data/train-volume.tif"
    TRAIN_LABEL = "data/train-labels.tif"
    full_dataset = EMDataset(
        TRAIN_VOLUME,
        TRAIN_LABEL
    )

    n = len(full_dataset)
    indices = np.random.permutation(n)
    VAL_RATIO = 0.2
    split = int(n * (1 - VAL_RATIO))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = EMDataset(
        TRAIN_VOLUME,
        TRAIN_LABEL,
        train_idx
    )
    val_ds = EMDataset(
        TRAIN_VOLUME,
        TRAIN_LABEL,
        val_idx
    )

    BATCH_SIZE = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ----- model -----
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_dice = 0.0
    EPOCHS = 50
    # ----- training loop -----
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(img)
            loss = bce_dice_loss(pred, mask)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # ----- validation -----
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                mask = mask.to(device)

                pred = model(img)
                val_dice += dice_score(pred, mask).item()
            
        val_dice /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Dice: {val_dice:.4f}"
        )

        # ----- save best -----
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_unet.pth")
            print("  ✓ Saved best model")

    print("Training done. Best Val Dice:", best_val_dice)


if __name__ == "__main__":
    train()
