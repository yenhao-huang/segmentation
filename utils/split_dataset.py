import os
import shutil
import random

# Set paths
img_dir = 'data/carvana_dataset/train/train'
mask_dir = 'data/carvana_dataset/train_masks/train_masks'

train_img_out = 'data/carvana_dataset/train_split/train'
train_mask_out = 'data/carvana_dataset/train_split/train_masks'
val_img_out = 'data/carvana_dataset/eval_split/eval'
val_mask_out = 'data/carvana_dataset/eval_split/eval_masks'

# Build output directories
for d in [train_img_out, train_mask_out, val_img_out, val_mask_out]:
    os.makedirs(d, exist_ok=True)

# Get all image files
all_imgs = sorted(os.listdir(img_dir))
random.seed(42)
random.shuffle(all_imgs)

# Split dataset into training and validation sets
val_percent = 0.1
val_size = int(len(all_imgs) * val_percent)
train_imgs = all_imgs[val_size:]
val_imgs = all_imgs[:val_size]

# Copy files to respective directories
def copy_files(img_list, img_out_dir, mask_out_dir):
    for img_file in img_list:
        mask_file = img_file.replace('.jpg', '_mask.gif')

        shutil.copy(os.path.join(img_dir, img_file), os.path.join(img_out_dir, img_file))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(mask_out_dir, mask_file))

copy_files(train_imgs, train_img_out, train_mask_out)
copy_files(val_imgs, val_img_out, val_mask_out)

print(f"Train size: {len(train_imgs)} images")
print(f"Val size: {len(val_imgs)} images")
