import torch
from torch.utils.data import DataLoader
from torch import optim
import os

from training.dataset import BDD100KLaneDataset
from models.unet import UNet

# 🔴 CRITICAL for Windows CPU
torch.set_num_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-4

IMG_DIR = "data/bdd100k/images/train"
MASK_DIR = "data/bdd100k/masks/train"

os.makedirs("checkpoints", exist_ok=True)


def main():
    dataset = BDD100KLaneDataset(
        IMG_DIR,
        MASK_DIR,
        img_size=(64, 128)
    )

    print("Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = UNet().to(DEVICE)

    # Sanity forward pass
    imgs, masks = next(iter(loader))
    imgs = imgs.to(DEVICE)
    out = model(imgs)
    print("Forward pass OK:", out.shape)

    bce = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = bce(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Epoch {epoch+1} | Batch {i}")

            # 🔴 HARD STOP after 20 batches
            if i == 20:
                break

        print(f"Epoch {epoch+1} done | Loss: {epoch_loss/(i+1):.4f}")

        torch.save(
            model.state_dict(),
            f"checkpoints/unet_epoch_{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
