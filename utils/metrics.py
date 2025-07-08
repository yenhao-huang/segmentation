import torch

def iou_score(pred, target, smooth=1e-6):
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Binarize the prediction: pixels > 0.5 -> 1, else 0
    pred = (pred > 0.5).float()
    target = target.float()

    # Compute the intersection where both pred and target are 1
    intersection = (pred * target).sum(dim=(1, 2, 3))

    # Compute the union where pred or target is 1
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3))

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()

def pixel_accuracy(pred, target):
    # Convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Binarize prediction (threshold = 0.5)
    pred = (pred > 0.5).float()
    
    target = target.float()

    # Total correct pixels
    correct = (pred == target).float().sum()

    # Total pixels
    total = torch.numel(pred)

    acc = correct / total
    return acc.item()
