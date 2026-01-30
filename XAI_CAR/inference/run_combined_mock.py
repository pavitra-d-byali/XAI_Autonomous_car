import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os

from models.unet import UNet

# ---------------- CONFIG ----------------
DEVICE = "cpu"
IMG_SIZE = (64, 128)   # must match your CPU training
MODEL_PATH = "checkpoints/unet_epoch_2.pth"

INPUT_DIR = "videos/input"
OUTPUT_DIR = "videos/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- MODEL ----------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor()
])

# ---------------- PICK ONE VIDEO ----------------
video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]
if len(video_files) == 0:
    raise RuntimeError("No videos found in videos/input")

video_path = os.path.join(INPUT_DIR, video_files[0])
output_path = os.path.join(OUTPUT_DIR, "combined_mock_output.mp4")

# ---------------- VIDEO IO ----------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing video:", video_path)

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inp = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(inp)

    mask = pred.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height))

    overlay = frame.copy()
    overlay[mask > 0] = [0, 0, 255]  # red lanes

    out.write(overlay)

    if frame_id % 30 == 0:
        print(f"Processed {frame_id} frames")

cap.release()
out.release()

print("✅ Video inference complete")
print("Saved to:", output_path)