import os
import imageio
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(512, 512)):
        # sorted to ensure consistent order btn images and masks
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = imageio.v2.imread(self.mask_paths[idx])
        if mask.ndim == 3:
            mask = mask[:, :, 0] # Convert to grayscale if mask is RGB

        img = cv2.resize(img, self.img_size) / 255.0 # Normalize to [0, 1]
        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)

        img = img.transpose(2, 0, 1).astype(np.float32) # Change to (C, H, W)
        mask = np.expand_dims(mask, 0)

        return torch.tensor(img), torch.tensor(mask)
