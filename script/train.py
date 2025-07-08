import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.carvana_dataset import CarvanaDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.unet import UNet
from utils.loss import dice_loss

# Paths
train_img_dir = 'data/carvana_dataset/train_split/train'
train_mask_dir = 'data/carvana_dataset/train_split/train_masks'
eval_img_dir = 'data/carvana_dataset/eval_split/eval'
eval_mask_dir = 'data/carvana_dataset/eval_split/eval_masks'

# Create dataset
train_dataset = CarvanaDataset(train_img_dir, train_mask_dir)
val_dataset = CarvanaDataset(eval_img_dir, eval_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(1, 11):  # Make epochs 1-indexed for display
    model.train()
    epoch_loss = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    for imgs, masks in train_pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        avg_loss = epoch_loss / (train_pbar.n + 1)
        train_pbar.set_postfix(loss=avg_loss)

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch}: Train Dice Loss = {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", unit="batch")
    with torch.no_grad():
        for imgs, masks in val_pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = dice_loss(preds, masks)
            val_loss += loss.item()

            avg_val = val_loss / (val_pbar.n + 1)
            val_pbar.set_postfix(val_loss=avg_val)

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}: Val Dice Loss = {avg_val_loss:.4f}")


torch.save(model.state_dict(), f"ckpt/model_epoch_{epoch}.pth")
print(f"Model saved: model_epoch_{epoch}.pth")