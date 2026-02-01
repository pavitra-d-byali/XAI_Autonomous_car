# inference/run_xai_video.py

import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os

from models.unet import UNet

# ---------- CONFIG ----------
DEVICE = "cpu"
IMG_SIZE = (64, 128)
MODEL_PATH = "checkpoints/unet_epoch_2.pth"

INPUT_VIDEO = "videos/input/nD_1.mp4"
OUTPUT_VIDEO = "videos/output/xai_lane_direction_video.mp4"

os.makedirs("videos/output", exist_ok=True)

# ---------- DIRECTION ESTIMATION ----------
def estimate_direction(mask):
    h, w = mask.shape
    bottom = mask[int(h * 0.75):, :]

    xs = np.where(bottom > 0)[1]
    if len(xs) == 0:
        return "UNKNOWN"

    lane_center = xs.mean()
    image_center = w / 2
    offset = lane_center - image_center

    if offset < -30:
        return "LEFT"
    elif offset > 30:
        return "RIGHT"
    else:
        return "STRAIGHT"

# ---------- MODEL ----------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor()
])

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ---------- PREP ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    inp = transform(pil).unsqueeze(0).to(DEVICE)
    inp.requires_grad = True

    # ---------- FORWARD ----------
    output = model(inp)
    mask_pred = output.squeeze().detach().cpu().numpy()
    mask_bin = (mask_pred > 0.5).astype(np.uint8) * 255
    mask_bin = cv2.resize(mask_bin, (w, h))

    # ---------- XAI ----------
    score = output.mean()
    score.backward()

    saliency = inp.grad.abs().max(dim=1)[0]
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    saliency = cv2.resize(saliency, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * saliency), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # ---------- DIRECTION ----------
    direction = estimate_direction(mask_bin)

    arrow_start = (w // 2, h - 50)
    arrow_end = {
        "LEFT": (w // 2 - 120, h - 120),
        "RIGHT": (w // 2 + 120, h - 120),
        "STRAIGHT": (w // 2, h - 140),
    }.get(direction, arrow_start)

    cv2.arrowedLine(
        overlay, arrow_start, arrow_end,
        (0, 255, 0), 5, tipLength=0.3
    )

    cv2.putText(
        overlay,
        f"Direction: {direction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    out.write(overlay)

    if frame_id % 30 == 0:
        print(f"Processed {frame_id} frames")

cap.release()
out.release()

print("✅ XAI + Direction video saved to:", OUTPUT_VIDEO)