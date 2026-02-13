import torch
import cv2
import numpy as np
from training.lane_unet import LaneUNet
from utils import preprocess_for_cnn

MODEL_PATH = "models/checkpoint/lane_cnn_weights.pth"
VIDEO_PATH = "videos/input/nD_2.mp4"

device = "cpu"

model = LaneUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read frame")

inp = preprocess_for_cnn(frame, size=(256, 512))
inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(inp)
    probs = torch.sigmoid(logits)

mask = probs.squeeze().cpu().numpy()

print("Min:", mask.min())
print("Max:", mask.max())
print("Mean:", mask.mean())

cv2.imshow("Raw Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
 