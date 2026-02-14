import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import optim

from models.architectures.unet import UNet
from src.lane_detection.bdd100k_dataset import BDD100KLaneDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)

EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = (128, 256)

# 🔥 FIX THIS PATH PROPERLY
IMG_DIR = "../data/lane_dataset/train/images"
MASK_DIR = "../data/lane_dataset/train/masks"

CHECKPOINT_DIR = "models/weights/lane"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():

    dataset = BDD100KLaneDataset(
        IMG_DIR,
        MASK_DIR,
        size=IMG_SIZE
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = UNet().to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "lane_unet_best.pth")
            )
            print("✅ Saved best model")

    print("Training complete.")


if __name__ == "__main__":
    main()
