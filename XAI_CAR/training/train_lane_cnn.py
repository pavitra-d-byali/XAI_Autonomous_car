import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----------------------------
# Lane CNN (Segmentation)
# ----------------------------
class LaneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # (B,1,H,W)


# ----------------------------
# Dummy Dataset (for now)
# ----------------------------
def load_dummy_data(num_samples=50, size=(224, 224)):
    images = []
    masks = []

    for _ in range(num_samples):
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        mask = np.zeros((size[0], size[1]), dtype=np.uint8)

        # draw fake lane
        cv2.line(mask, (110, 224), (140, 0), 255, 5)

        img = img / 255.0
        mask = mask / 255.0

        images.append(img.transpose(2, 0, 1))
        masks.append(mask[np.newaxis, :, :])

    return torch.tensor(images, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)


# ----------------------------
# Training Loop
# ----------------------------
def train():
    device = "cpu"
    model = LaneCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    images, masks = load_dummy_data()
    images, masks = images.to(device), masks.to(device)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

    # Save weights ONLY (PyTorch 2.6 safe)
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/lane_cnn_weights.pth")

    print("lane_cnn_weights.pth saved successfully")


if __name__ == "__main__":
    train()
