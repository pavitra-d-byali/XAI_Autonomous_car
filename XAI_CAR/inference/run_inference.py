import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from models.unet import UNet

# ---- CONFIG ----
DEVICE = "cpu"
IMG_SIZE = (64, 128)

MODEL_PATH = "checkpoints/unet_epoch_2.pth"
TEST_IMAGE = "data/bdd100k/images/train/75fcb559-4cdb2554.jpg"

OUTPUT_DIR = "inference/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load model ----
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- Image transform ----
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor()
])

# ---- Load image ----
image = Image.open(TEST_IMAGE).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ---- Inference ----
with torch.no_grad():
    pred = model(input_tensor)

# ---- Post-process ----
pred_mask = pred.squeeze().cpu().numpy()
binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

# ---- Resize back for visualization ----
image_small = image.resize((IMG_SIZE[1], IMG_SIZE[0]))

# ---- Save outputs ----
Image.fromarray(binary_mask).save(f"{OUTPUT_DIR}/mask.png")

overlay = np.array(image_small)
overlay[binary_mask > 0] = [255, 0, 0]  # red lane

Image.fromarray(overlay).save(f"{OUTPUT_DIR}/overlay.png")

image_small.save(f"{OUTPUT_DIR}/original.png")

print("✅ Inference complete")
print("Saved files:")
print(" - inference/output/original.png")
print(" - inference/output/mask.png")
print(" - inference/output/overlay.png")
