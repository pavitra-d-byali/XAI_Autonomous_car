import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode




class BDD100KLaneDataset(Dataset):
    """
    BDD100K Lane Segmentation Dataset

    Returns: 
        image: Tensor [3, H, W]
        mask : Tensor [1, H, W] (binary)
    """

    def __init__(self, img_dir, mask_dir, img_size=(256, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        if not os.path.exists(img_dir):
            raise RuntimeError(f"Image directory not found: {img_dir}")

        if not os.path.exists(mask_dir):
            raise RuntimeError(f"Mask directory not found: {mask_dir}")

        mask_files = set(os.listdir(mask_dir))

        # Keep only valid image-mask pairs
        self.images = [
            img for img in os.listdir(img_dir)
            if img.endswith(".jpg")
            and img.replace(".jpg", ".png") in mask_files
        ]

        self.images.sort()

        if len(self.images) == 0:
            raise RuntimeError("No valid image-mask pairs found!")

        print(f"[INFO] Found {len(self.images)} valid training pairs")

        # Image transform (bilinear is fine)
        self.img_transform = T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

        # Mask transform (NEAREST is critical)
        self.mask_resize = T.Resize(
            img_size,
            interpolation=InterpolationMode.NEAREST
        )

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

        mask = self.mask_resize(mask)
        mask = TF.to_tensor(mask)

        # Convert to binary
        mask = (mask > 0.5).float()

        return image, mask
