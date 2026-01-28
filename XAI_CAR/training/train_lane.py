import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.bdd100k_dataset import BDD100KLaneDataset
from models.lane_unet import LaneUNet
import inspect
from training.bdd100k_dataset import BDD100KLaneDataset

print("USING DATASET FROM:", inspect.getfile(BDD100KLaneDataset))



# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =========================
# Dataset & DataLoader
# =========================
train_dataset = BDD100KLaneDataset(
    images_dir="data/bdd100k/images/train",
    masks_dir="data/bdd100k/masks/train",
    size=(120, 120)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,          # ✅ REQUIRED on Windows
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=True          # ✅ avoids small last batch instability
)


# =========================
# Model, Loss, Optimizer
# =========================
model = LaneUNet().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# =========================
# Training Loop
# =========================
EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        preds = model(images)
        loss = criterion(preds, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {avg_loss:.4f}")


# =========================
# Save model
# =========================
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "epochs": EPOCHS,
        "loss": avg_loss,
    },
    "models/lane_unet.pth"
)

print("✅ Model saved to models/lane_unet.pth")

# minor refactor note
