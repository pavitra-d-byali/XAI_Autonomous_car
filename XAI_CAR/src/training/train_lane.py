import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim

from src.training.dataset import BDD100KLaneDataset
from src.training.lane_unet import LaneUNet


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# DEVICE
# ============================================================
torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)


# ============================================================
# CONFIG
# ============================================================
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE =  (128, 256)


IMG_DIR = "data/bdd100k/images/train"
MASK_DIR = "data/bdd100k/masks/train"

CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# MAIN
# ============================================================
def main():

    # ---------------- Dataset ----------------
    full_dataset = BDD100KLaneDataset(
        IMG_DIR,
        MASK_DIR,
        img_size=IMG_SIZE
    )

    print("[INFO] Total dataset size:", len(full_dataset))

    # 80 / 20 split (since val masks are empty)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("[INFO] Train size:", len(train_dataset))
    print("[INFO] Val size:", len(val_dataset))

    # ---------------- Model ----------------
    model = LaneUNet().to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------------- Sanity Forward Pass ----------------
    imgs, masks = next(iter(train_loader))
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        out = model(imgs)
    print("[INFO] Forward pass OK:", out.shape)

    # ========================================================
    # TRAINING LOOP
    # ========================================================
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

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

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # ---------------- Save Best Model ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(
                CHECKPOINT_DIR,
                "lane_unet_best.pth"
            )
            torch.save(model.state_dict(), best_path)
            print("[INFO] Saved BEST model")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
