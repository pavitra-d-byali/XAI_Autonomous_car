import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BDD100KLaneDataset(Dataset):
    """
    Binary lane segmentation dataset.
    Assumes:
    - Images: .jpg
    - Masks:  .png (0 background, 255 lane)
    """

    def __init__(self, images_dir, masks_dir, size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.size = size

        self.samples = []

        for img_name in sorted(os.listdir(images_dir)):
            if not img_name.endswith(".jpg"):
                continue

            base = os.path.splitext(img_name)[0]
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, base + ".png")

            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("❌ No valid image-mask pairs found")

        print(f"✅ Loaded {len(self.samples)} valid image-mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"❌ Failed to load image: {img_path}")

        if mask is None or mask.size == 0:
            raise RuntimeError(f"❌ Failed to load mask: {mask_path}")

        # Resize
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Binary mask (0 or 1)
        mask = (mask > 0).astype(np.float32)

        # To tensors
        image = torch.from_numpy(image.transpose(2, 0, 1))   # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)            # (1, H, W)

        return image, mask
