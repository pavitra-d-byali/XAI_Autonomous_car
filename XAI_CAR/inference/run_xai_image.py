# inference/run_xai_image.py

import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import os

from models.unet import UNet

# ---------- CONFIG ----------
DEVICE = "cpu"
IMG_SIZE = (64, 128)
MODEL_PATH = "checkpoints/unet_epoch_2.pth"

IMAGE_PATH = "data/bdd100k/images/val/0001.jpg"  # change if needed
OUTPUT_PATH = "inference/output/xai_image.png"

os.makedirs("inference/output", exist_ok=True)

# ---------- LOAD MODEL ----------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor()
])

# ---------- LOAD IMAGE ----------
img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)
input_tensor.requires_grad = True

# ---------- FORWARD ----------
output = model(input_tensor)
score = output.mean()          # aggregate lane activation
score.backward()

# ---------- SALIENCY ----------
saliency = input_tensor.grad.abs().max(dim=1)[0]
saliency = saliency.squeeze().cpu().numpy()

saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
saliency = cv2.resize(saliency, img.size)

heatmap = cv2.applyColorMap(
    np.uint8(255 * saliency), cv2.COLORMAP_JET
)

original = np.array(img)
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

cv2.imwrite(OUTPUT_PATH, overlay)

print("✅ XAI image saved to:", OUTPUT_PATH)