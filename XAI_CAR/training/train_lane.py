import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bdd100k_dataset import BDD100KLaneDataset
from lane_unet import LaneUNet

# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================
# Dataset & DataLoader
# =========================
train_dataset = BDD100KLaneDataset(
    images_dir="data/bdd100k/images/train",   # ✅ CORRECT NAME
    masks_dir="data/bdd100k/masks/train"     # ✅ CORRECT NAME
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,      # ✅ REQUIRED ON WINDOWS
    pin_memory=False
)

# =========================
# Model
# =========================
model = LaneUNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# Training Loop
# =========================
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss / len(train_loader):.4f}")

# =========================
# Save model
# =========================
torch.save(model.state_dict(), "models/lane_unet.pth")
print("✅ Model saved: models/lane_unet.pth")
