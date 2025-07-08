import sys
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.carvana_dataset import CarvanaDataset
from model.unet import UNet
from utils.metrics import iou_score, pixel_accuracy

def visualize_predictions(imgs, masks, preds, batch_idx, save_dir="preds"):
    """
    Save side-by-side visualizations of images, ground truth masks, and predicted masks.

    Args:
        imgs (Tensor): Batch of input images [B, C, H, W]
        masks (Tensor): Ground truth masks [B, 1, H, W]
        preds (Tensor): Raw model predictions [B, 1, H, W]
        batch_idx (int): Index of the current batch
        save_dir (str): Directory to save output images
    """
    os.makedirs(save_dir, exist_ok=True)
    preds_bin = (torch.sigmoid(preds) > 0.5).float()

    for i in range(imgs.size(0)):
        img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
        mask_np = masks[i].squeeze().cpu().numpy()
        pred_np = preds_bin[i].squeeze().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img_np)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].axis("off")

        ax[2].imshow(pred_np, cmap="gray")
        ax[2].set_title("Prediction")
        ax[2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"pred_{batch_idx}_{i}.png"))
        plt.close()

# Dataset paths
eval_img_dir = 'data/carvana_dataset/eval_split/eval'
eval_mask_dir = 'data/carvana_dataset/eval_split/eval_masks'

# Load dataset
val_dataset = CarvanaDataset(eval_img_dir, eval_mask_dir)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet().to(device)
model.load_state_dict(torch.load("ckpt/model_epoch_10.pth", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.eval()

# Evaluation loop
total_iou, total_acc = 0, 0
num_batches = 0

with torch.no_grad():
    for imgs, masks in tqdm(val_loader, desc="Eval", unit="batch"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)

        # Compute metrics
        iou = iou_score(preds, masks)
        acc = pixel_accuracy(preds, masks)
        total_iou += iou
        total_acc += acc
        num_batches += 1

        # Save visualizations
        visualize_predictions(imgs, masks, preds, num_batches)

# Final result
avg_iou = total_iou / num_batches
print(f"✅ Val IoU = {avg_iou:.4f}")
print(f"✅ Val Pixel Accuracy = {total_acc / num_batches:.4f}")