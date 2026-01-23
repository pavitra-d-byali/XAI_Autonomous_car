import os
import json
import cv2
import numpy as np

# ---------------- CONFIG ----------------
TRAIN_JSON = "data/bdd100k/labels/lane/train.json"
VAL_JSON   = "data/bdd100k/labels/lane/val.json"

TRAIN_IMG_DIR = "data/bdd100k/images/train"
VAL_IMG_DIR   = "data/bdd100k/images/val"

TRAIN_MASK_DIR = "data/bdd100k/masks/train"
VAL_MASK_DIR   = "data/bdd100k/masks/val"

IMG_H, IMG_W = 720, 1280
# ----------------------------------------

os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
os.makedirs(VAL_MASK_DIR, exist_ok=True)

def process_split(json_path, img_dir, out_dir):
    print(f"Processing {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    for item in data:
        img_name = item["name"]
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        for lane in item.get("labels", []):
            if lane["category"] != "lane":
                continue

            for poly in lane["poly2d"]:
                pts = poly["vertices"]
                pts = np.array(pts, np.int32)
                cv2.polylines(mask, [pts], False, 255, thickness=5)

        out_path = os.path.join(out_dir, img_name.replace(".jpg", ".png"))
        cv2.imwrite(out_path, mask)

        print("Saved:", out_path)

process_split(TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_MASK_DIR)
process_split(VAL_JSON,   VAL_IMG_DIR,   VAL_MASK_DIR)

print("✅ Mask generation complete")
