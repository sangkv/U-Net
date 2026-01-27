import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


@torch.no_grad()
def dice_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection) / (union + 1e-8)
