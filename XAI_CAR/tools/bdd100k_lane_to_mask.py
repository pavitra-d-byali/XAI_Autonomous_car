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
# ----------------------------------------

os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
os.makedirs(VAL_MASK_DIR, exist_ok=True)


def generate_mask(item, img_dir, out_dir, available_images):
    img_name = item.get("name")

    if img_name not in available_images:
        return False

    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)

    if image is None:
        return False

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    labels = item.get("labels", [])

    for lane in labels:
        category = lane.get("category", "")

        if "lane" not in category:
            continue

        for poly in lane.get("poly2d", []):
            vertices = poly.get("vertices", [])
            if len(vertices) < 2:
                continue

            pts = np.array(vertices, dtype=np.int32)
            cv2.polylines(mask, [pts], False, 255, thickness=12)

    # 🔥 Dilation to make lanes stronger
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    out_path = os.path.join(out_dir, img_name.replace(".jpg", ".png"))
    cv2.imwrite(out_path, mask)

    return True


def process_split(json_path, img_dir, out_dir):
    print(f"\nProcessing: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    available_images = set(os.listdir(img_dir))

    saved = 0
    skipped = 0

    for item in data:
        success = generate_mask(item, img_dir, out_dir, available_images)
        if success:
            saved += 1
        else:
            skipped += 1

    print(f"Saved masks: {saved}")
    print(f"Skipped items: {skipped}")
    print("Finished.")


process_split(TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_MASK_DIR)
process_split(VAL_JSON,   VAL_IMG_DIR,   VAL_MASK_DIR)

print("\n✅ Mask generation complete")
 