import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BDD100KLaneDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(64, 128)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        # Keep only valid image-mask pairs
        mask_files = set(os.listdir(mask_dir))

        self.images = [
            img for img in os.listdir(img_dir)
            if img.endswith(".jpg") and img.replace(".jpg", ".png") in mask_files
        ]
        self.images.sort()

        # 🔴 NO normalization (CPU-safe)
        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir,
            img_name.replace(".jpg", ".png")
        )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # Binary mask
        mask = (mask > 0.5).float()

        return image, mask
