import torch
from torch.utils.data import DataLoader
from unet import UNet
from dataset import EMSegmentationDataset
from losses import bce_dice_loss
from utils import save_checkpoint, dice_score


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)

    dataset = EMSegmentationDataset(
        "data/train-volume.tif",
        "data/train-labels.tif"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_dice = 0.0

        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = bce_dice_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_score(pred, mask).item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss/len(loader):.4f} "
            f"Dice: {epoch_dice/len(loader):.4f}"
        )

    save_checkpoint(model, "unet_em.pth")


if __name__ == "__main__":
    train()
